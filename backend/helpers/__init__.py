from __future__ import annotations

"""
helpers 패키지 초기화 모듈.

목표
- 순환 임포트(circular import) 방지: 런타임에 내부 모듈을 재노출하지 않음.
- import * 금지: __all__ 을 빈 리스트로 유지.
- 타입 힌트 편의: TYPE_CHECKING 블록에서만 내부 심볼을 노출(런타임 영향 없음).

주의
- 여기서는 어떤 부작용(side effect)도 발생시키지 않습니다.
- 로깅/환경 로딩 등은 각 모듈(utils.py 등)에서 필요 시 명시적으로 수행하세요.
"""

from typing import TYPE_CHECKING

# 패키지 메타데이터(선택 사항)
__pkg_name__ = "helpers"
__version__ = "2025.08.12"

# from helpers import * 를 의도적으로 비활성화
__all__: list[str] = []

if TYPE_CHECKING:
    # 아래 임포트는 타입체킹 전용입니다(런타임에는 실행되지 않음).
    # 패키지 외부에서 IDE 자동완성/정적 분석에 도움을 줍니다.
    from .signals import generate_signal, manage_trade  # noqa: F401
    from .binance_client import get_overview  # noqa: F401
