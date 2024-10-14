# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# 이 코드는 EleutherAI의 GPT-NeoX 라이브러리와 해당 라이브러리 내의 GPT-NeoX 및 OPT 구현을 기반으로 합니다.
# Meta AI 팀이 훈련한 모델에 사용된 GPT-NeoX와 OPT와 비교하여 사소한 구조적 차이를 수용하기 위해 원래 형태에서 수정되었습니다.
#
# Apache License, Version 2.0 ("License")에 따라 라이선스가 부여됩니다.
# 이 파일을 사용하려면 라이선스에 따라 사용해야 합니다.
# 라이선스 사본은 다음에서 찾을 수 있습니다:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 해당 법률에서 요구되거나 서면으로 합의되지 않는 한, 소프트웨어는 "있는 그대로" 배포되며,
# 명시적이든 묵시적이든 어떠한 종류의 보증도 제공되지 않습니다.
# 라이선스에 따른 권한과 제한 사항은 라이선스를 참조하십시오.
import math
from typing import List, Optional, Tuple, Union

import torch  # PyTorch를 임포트하여 딥러닝 연산을 수행합니다.
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn  # 신경망 모듈을 임포트합니다.

from transformers.activations import (
    ACT2FN,
)  # 다양한 활성화 함수들을 함수 이름과 연결해주는 매핑 테이블입니다.

from transformers.cache_utils import Cache, DynamicCache, StaticCache

# 모델 실행 중에 캐시를 효율적으로 관리하기 위한 유틸리티입니다.
# Cache: 일반 캐시 클래스,
# DynamicCache: 동적 캐시 관리,
# StaticCache: 정적 캐시 관리

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,  # 기본 모델 출력 구조로, 이전 토큰 히스토리를 포함합니다.
    CausalLMOutputWithPast,  # 언어 모델링 작업에서 이전 토큰 히스토리와 함께 출력되는 구조입니다.
    QuestionAnsweringModelOutput,  # 질문-답변 태스크를 위한 모델 출력 구조입니다.
    SequenceClassifierOutputWithPast,  # 시퀀스 분류 작업에서 이전 히스토리를 포함한 출력 구조입니다.
    TokenClassifierOutput,  # 토큰 분류 작업을 위한 모델 출력 구조입니다.
)

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# 상대적 위치 인코딩(Relative Position Encoding)을 처리하는 함수들입니다.

from transformers.modeling_utils import PreTrainedModel

# 모든 사전 학습된 모델의 기반이 되는 클래스입니다. 다양한 공통 기능을 제공합니다.

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# 레이어 정규화에 사용하는 모든 레이어 종류를 포함한 상수로, 레이어 정규화를 처리할 때 사용됩니다.

from transformers.utils import (
    add_start_docstrings,  # 모델 클래스와 함수에 설명을 추가하는 데 사용됩니다.
    add_start_docstrings_to_model_forward,  # 모델의 `forward` 메서드에 설명을 추가하는 데 사용됩니다.
    is_flash_attn_greater_or_equal_2_10,  # 플래시 어텐션의 버전이 2.10 이상인지 확인하는 유틸리티 함수입니다.
    logging,  # 모델 실행 과정에서 로그 출력을 관리하는 데 사용됩니다.
    replace_return_docstrings,  # 리턴 값에 대한 설명을 대체하는 유틸리티입니다.
)

from configuration_llama import LlamaConfig

# LLaMA 모델의 설정 클래스로, 모델의 하이퍼파라미터와 구조 설정을 관리합니다.

logger = logging.get_logger(__name__)
# 로거(logger)를 설정하여 로그를 기록합니다.

_CONFIG_FOR_DOC = "LlamaConfig"  # 문서화를 위한 설정 이름입니다.


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm은 T5LayerNorm과 동일한 역할을 하는 RMSNorm 레이어입니다.

        파라미터:
        - hidden_size (int): 히든 사이즈, 첫번째 레이어에서는 임베딩 벡터의 차원이고 이후 레이어에서는 이전 레이어의 출력이 됩니다.
        - eps (float): 분산 계산 시 0으로 나누는 것을 방지하기 위한 작은 값입니다.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )  # 학습 가능한 가중치 파라미터입니다.
        self.variance_epsilon = eps  # 분산 계산 시 사용되는 작은 값입니다.

    def forward(self, hidden_states):
        """
        입력된 히든 스테이트에 RMS 노름을 적용합니다.

        파라미터:
        - hidden_states (torch.Tensor): 입력 텐서로, 형태는 (배치 크기, 시퀀스 길이, 히든 크기)입니다.

        반환값:
        - torch.Tensor: RMS 노름이 적용된 텐서입니다.
        """
        input_dtype = hidden_states.dtype  # 입력의 데이터 타입을 저장합니다.
        hidden_states = hidden_states.to(
            torch.float32
        )  # float32 타입으로 변환하여 수치적 안정성을 확보하기 위해 연산 중에는 더 높은 정밀도로 계산합니다.
        variance = hidden_states.pow(2).mean(
            -1, keepdim=True
        )  # 마지막 차원에 대한 분산을 계산합니다.
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )  # 정규화합니다.
        return self.weight * hidden_states.to(
            input_dtype
        )  # 메모리와 성능을 효율화하기 위해 다시 원래의 데이터 형식으로 변환하여 가중치와 곱합니다.

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"  # 레이어의 추가 정보를 문자열로 반환합니다.


# RMSNorm 레이어를 모든 LayerNorm 레이어의 리스트에 추가합니다.
ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        """
        RoPE(Rotary Positional Embedding)를 구현하는 클래스입니다.

        파라미터:
        - dim (int): 임베딩 차원 수입니다.
        - max_position_embeddings (int): 최대 포지션 임베딩 길이입니다.
        - base (int): 주파수 스케일링을 위한 값으로 작은 값을 사용하면 ₩
        - device (torch.device): 연산이 수행될 디바이스입니다.
        - scaling_factor (float): 스케일링 팩터입니다.
        - rope_type (str): RoPE의 유형입니다.
        - config (LlamaConfig): 모델 설정입니다.
        """
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            # 호환성을 위한 코드이며, 향후 버전에서는 제거될 예정입니다.
            logger.warning_once(
                "`LlamaRotaryEmbedding`은 이제 모델 설정을 `config` 인자를 통해 전달하여 완전히 매개변수화될 수 있습니다. "
                "v4.46에서는 다른 인자들이 제거될 예정입니다."
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # 호환성을 위한 코드입니다.
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # RoPE 초기화를 수행합니다.
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )  # 역주파수 값을 버퍼로 저장합니다.
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        동적 RoPE 레이어에서 역주파수(inv_freq)를 업데이트하는 함수입니다.

        파라미터:
        - position_ids (torch.Tensor): 포지션 아이디 텐서입니다.
        - device (torch.device): 연산이 수행되는 디바이스입니다.
        """
        seq_len = torch.max(position_ids) + 1  # 현재 시퀀스 길이를 계산합니다.
        if seq_len > self.max_seq_len_cached:
            # 시퀀스 길이가 캐시된 최대 길이를 초과하면 역주파수를 재계산합니다.
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        # 시퀀스 길이가 원래 최대 길이보다 작아지면 원래 역주파수로 복원합니다.
        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        RoPE를 적용하기 위한 cos와 sin 값을 계산합니다.

        파라미터:
        - x (torch.Tensor): 입력 텐서입니다.
        - position_ids (torch.Tensor): 포지션 아이디 텐서입니다.

        반환값:
        - Tuple[torch.Tensor, torch.Tensor]: cos와 sin 텐서입니다.
        """
        if "dynamic" in self.rope_type:
            # 동적 RoPE의 경우 역주파수를 업데이트합니다.
            self._dynamic_frequency_update(position_ids, device=x.device)

        # RoPE의 핵심 계산 부분입니다.
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # float32로 강제 변환하여 연산의 안정성을 확보합니다.
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # 고급 RoPE 유형의 경우 스케일링 팩터를 적용합니다.
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    입력 텐서의 절반 차원에 음수를 곱하여 회전 변환을 수행합니다.

    파라미터:
    - x (torch.Tensor): 입력 텐서입니다.

    반환값:
    - torch.Tensor: 회전 변환이 적용된 텐서입니다.
    """
    x1 = x[..., : x.shape[-1] // 2]  # 앞쪽 절반을 분리합니다.
    x2 = x[..., x.shape[-1] // 2 :]  # 뒤쪽 절반을 분리합니다.
    return torch.cat((-x2, x1), dim=-1)  # 뒤쪽 절반에 음수를 곱하고 다시 결합합니다.


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    쿼리(q)와 키(k) 텐서에 RoPE를 적용합니다.

    파라미터:
    - q (torch.Tensor): 쿼리 텐서입니다.
    - k (torch.Tensor): 키 텐서입니다.
    - cos (torch.Tensor): cos 값을 가진 텐서입니다.
    - sin (torch.Tensor): sin 값을 가진 텐서입니다.
    - position_ids (torch.Tensor, optional): 포지션 아이디 텐서입니다. (사용되지 않음)
    - unsqueeze_dim (int): 차원을 확장할 위치입니다.

    반환값:
    - Tuple[torch.Tensor, torch.Tensor]: RoPE가 적용된 쿼리와 키 텐서입니다.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # 지정된 차원에 새로운 차원을 추가합니다.
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 쿼리에 RoPE를 적용합니다.
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 키에 RoPE를 적용합니다.
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        """
        Llama 모델의 MLP(Feed-Forward) 부분을 구현한 클래스입니다.

        파라미터:
        - config (LlamaConfig): 모델 설정입니다.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 히든 크기입니다.
        self.intermediate_size = config.intermediate_size  # 인터미디어트 크기입니다.
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )  # 게이트 프로젝션 레이어입니다.
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )  # 업 프로젝션 레이어입니다.
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )  # 다운 프로젝션 레이어입니다.
        self.act_fn = ACT2FN[config.hidden_act]  # 활성화 함수입니다.

    def forward(self, x):
        """
        입력 텐서 x에 대해 MLP 연산을 수행합니다.

        파라미터:
        - x (torch.Tensor): 입력 텐서입니다.

        반환값:
        - torch.Tensor: MLP 연산 결과 텐서입니다.
        """
        if self.config.pretraining_tp > 1:
            # 사전 훈련 시 텐서 병렬 처리를 위한 코드입니다.
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # 일반적인 MLP 연산입니다.
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    키와 값 텐서를 반복하여 다중 헤드에 맞게 확장합니다.

    파라미터:
    - hidden_states (torch.Tensor): 입력 키 또는 값 텐서입니다. 형태는 (배치 크기, num_key_value_heads, 시퀀스 길이, 헤드 크기)입니다.
    - n_rep (int): 반복 횟수입니다. num_attention_heads // num_key_value_heads와 같습니다.

    반환값:
    - torch.Tensor: 확장된 텐서로, 형태는 (배치 크기, num_attention_heads, 시퀀스 길이, 헤드 크기)입니다.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 텐서를 확장하여 반복합니다.
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """
    "Attention Is All You Need" 논문의 멀티헤드 어텐션을 구현한 클래스입니다.
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        """
        파라미터:
        - config (LlamaConfig): 모델 설정입니다.
        - layer_idx (int, optional): 레이어 인덱스입니다.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"`layer_idx`를 지정하지 않고 {self.__class__.__name__}를 인스턴스화하는 것은 권장되지 않으며, "
                "캐싱이 사용되는 경우 forward 호출 시 오류가 발생할 수 있습니다. "
                "이 클래스를 생성할 때 `layer_idx`를 제공해야 합니다."
            )

        self.attention_dropout = config.attention_dropout  # 어텐션 드롭아웃 비율입니다.
        self.hidden_size = config.hidden_size  # 히든 크기입니다.
        self.num_heads = config.num_attention_heads  # 어텐션 헤드의 수입니다.
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )  # 각 헤드의 크기입니다.
        self.num_key_value_heads = config.num_key_value_heads  # 키/값 헤드의 수입니다.
        self.num_key_value_groups = (
            self.num_heads // self.num_key_value_heads
        )  # 키/값 헤드 그룹의 수입니다.
        self.max_position_embeddings = (
            config.max_position_embeddings
        )  # 최대 포지션 임베딩 길이입니다.
        self.rope_theta = config.rope_theta  # RoPE의 스케일링 파라미터입니다.
        self.is_causal = True  # 캐주얼 마스크를 사용할지 여부입니다.

        # 쿼리, 키, 값, 출력 프로젝션 레이어를 정의합니다.
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # RoPE 임베딩을 초기화합니다.
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        어텐션 연산을 수행합니다.

        파라미터:
        - hidden_states (torch.Tensor): 입력 텐서입니다.
        - attention_mask (torch.Tensor, optional): 어텐션 마스크입니다.
        - position_ids (torch.LongTensor, optional): 포지션 아이디입니다.
        - past_key_value (Cache, optional): 이전의 키/값 캐시입니다.
        - output_attentions (bool): 어텐션 가중치를 출력할지 여부입니다.
        - use_cache (bool): 캐시를 사용할지 여부입니다.
        - cache_position (torch.LongTensor, optional): 캐시 위치입니다.
        - position_embeddings (Tuple[torch.Tensor, torch.Tensor], optional): 외부에서 계산된 cos와 sin 텐서입니다.

        반환값:
        - Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: 어텐션 출력, 어텐션 가중치, 키/값 캐시입니다.
        """
        bsz, q_len, _ = hidden_states.size()  # 배치 크기와 쿼리 길이를 얻습니다.

        if self.config.pretraining_tp > 1:
            # 사전 훈련 시 텐서 병렬 처리를 위한 코드입니다.
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            # 일반적인 프로젝션 연산입니다.
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 쿼리, 키, 값을 헤드 차원으로 분할합니다.
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            # RoPE 임베딩을 계산합니다.
            logger.warning_once(
                "이 모델의 어텐션 레이어는 `position_ids`를 통해 내부적으로 RoPE 임베딩을 계산하는 것에서 "
                "외부적으로 계산된 `position_embeddings`를 사용하는 것으로 전환하고 있습니다. "
                "v4.46에서는 `position_ids`가 제거되고 `position_embeddings`가 필수가 될 것입니다."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # 캐시된 키/값과 새로운 키/값을 결합합니다.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 키와 값을 반복하여 헤드 수에 맞춥니다.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 어텐션 가중치를 계산합니다.
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # 어텐션 마스크를 적용합니다.
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 어텐션 가중치에 소프트맥스를 적용합니다.
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)  # 가중합을 계산합니다.

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output`의 크기는 {(bsz, self.num_heads, q_len, self.head_dim)}이어야 하지만 "
                f"{attn_output.size()}입니다."
            )

        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # 원래 차원으로 변환합니다.
        attn_output = attn_output.reshape(bsz, q_len, -1)  # 출력 형태로 변환합니다.

        if self.config.pretraining_tp > 1:
            # 텐서 병렬 처리를 위한 출력 프로젝션입니다.
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            # 출력 프로젝션을 수행합니다.
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None  # 어텐션 가중치를 반환하지 않습니다.

        return (
            attn_output,
            attn_weights,
            past_key_value,
        )  # 어텐션 출력과 선택적으로 어텐션 가중치, 캐시를 반환합니다.
