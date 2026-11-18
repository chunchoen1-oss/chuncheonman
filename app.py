from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 0. 페이지 기본 설정 및 스타일
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="미국 ETF 레버리지 및 분산투자 분석 대시보드"
)


# ==========================================
# 1. 데이터 로딩 (캐싱)
# ==========================================
@st.cache_data
def load_data():
    """CSV 데이터를 읽어 날짜를 변환하고 종가 기준 피벗 테이블을 생성합니다."""
    csv_path = Path(__file__).parent / "data" / "etf_prices.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    pivot_df = df.pivot(index="datetime", columns="symbol", values="close")
    return pivot_df


# ==========================================
# 2. 금융 지표 공통 계산 함수 (화면 호출 분리)
# ==========================================
def calc_total_return(prices_series):
    """주어진 가격 시계열의 총수익률을 계산합니다."""
    return (prices_series.iloc[-1] / prices_series.iloc[0]) - 1.0


def calc_cagr(prices_series):
    """연환산수익률(CAGR)을 계산합니다. 연수 = 경과일수 / 365.25"""
    total_ret = calc_total_return(prices_series)
    years = (prices_series.index[-1] - prices_series.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((1.0 + total_ret) ** (1.0 / years)) - 1.0


def calc_volatility(prices_series):
    """일간수익률의 표준편차를 기반으로 연변동성을 계산합니다. (std * sqrt(252))"""
    daily_ret = prices_series.pct_change().dropna()
    return daily_ret.std(ddof=1) * np.sqrt(252)


def calc_drawdown_series(prices_series):
    """고점 대비 낙폭(Drawdown) 시계열을 계산합니다."""
    cummax = prices_series.cummax()
    return (prices_series / cummax) - 1.0


def calc_mdd(prices_series):
    """최대낙폭(MDD)을 계산합니다."""
    return calc_drawdown_series(prices_series).min()


def calc_portfolio_series(full_index, daily_ret_df, weight_spy, weight_gld):
    """일별 리밸런싱 가정 하에 두 자산 포트폴리오의 정규화 지수(시작=100)를 계산합니다."""
    p_daily_ret = (weight_spy * daily_ret_df["SPY"]) + (weight_gld * daily_ret_df["GLD"])
    p_prices = pd.Series(100.0, index=full_index)
    p_prices.iloc[1:] = 100.0 * (1.0 + p_daily_ret).cumprod()
    return p_prices


# ==========================================
# 3. 메인 앱 실행 흐름
# ==========================================
def main():
    try:
        prices_df = load_data()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.stop()

    min_date = prices_df.index.min().date()
    max_date = prices_df.index.max().date()

    # ------------------------------------------
    # 사이드바: 분석 기간 및 안내
    # ------------------------------------------
    st.sidebar.header("분석 설정")

    start_date = st.sidebar.date_input(
        "분석 시작일",
        value=pd.to_datetime("2021-08-23").date(),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.sidebar.date_input(
        "분석 종료일",
        value=pd.to_datetime("2026-08-20").date(),
        min_value=min_date,
        max_value=max_date,
    )

    # 배당 미포함 주의 문구 (상시 표시, 접거나 숨기지 않음)
    st.sidebar.markdown("---")
    st.sidebar.warning("주의: 종가 기준 가격 수익률이며 배당은 포함되지 않았습니다. 실제 성과와 다릅니다.")

    # 시작일/종료일 검증
    if start_date >= end_date:
        st.error("분석 시작일은 분석 종료일보다 앞서야 합니다. 날짜를 다시 설정해 주세요.")
        st.stop()

    # 데이터 슬라이싱
    sub_prices = prices_df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if sub_prices.empty:
        st.error("선택한 기간에 해당하는 데이터가 없습니다.")
        st.stop()

    # 60영업일 미만 경고
    if len(sub_prices) < 60:
        st.sidebar.warning("구간이 짧아 연환산 지표가 불안정합니다.")

    # 대시보드 헤더
    st.title("미국 ETF 레버리지 및 분산투자 분석")
    st.caption("Twelve Data 일별 가격 데이터(2021-08-23 ~ 2026-08-20) 기반 분석 대시보드")

    # 탭 생성
    tab1, tab2 = st.tabs(["레버리지의 함정", "분산투자의 힘"])

    # ==========================================
    # 탭 1 — 레버리지의 함정
    # ==========================================
    with tab1:
        st.header("레버리지 ETF의 장기 성과 및 변동성 위험")

        lev_symbols = ["QQQ", "QLD", "TQQQ"]
        lev_prices = sub_prices[lev_symbols]

        # 3-5. 핵심 메시지 자동 생성 및 출력
        qqq_tot = calc_total_return(lev_prices["QQQ"])
        tqqq_tot = calc_total_return(lev_prices["TQQQ"])
        tqqq_mdd = calc_mdd(lev_prices["TQQQ"])

        diff_ratio = (tqqq_tot / qqq_tot) if qqq_tot != 0 else 0.0

        if qqq_tot > 0:
            if diff_ratio < 1.0:
                summary_msg = f"배수는 3배이지만 수익률은 1배인 QQQ보다 낮았습니다. (1배 수익률의 {diff_ratio:.1f}배 수준)"
            else:
                summary_msg = f"배수는 3배이지만 수익률 차이는 {diff_ratio:.1f}배에 그쳤습니다."
        else:
            summary_msg = "기초자산(QQQ)이 하락하면서 레버리지 상품의 손실이 크게 확대되었습니다."

        msg = (
            f"{start_date} ~ {end_date} 구간에서 TQQQ(3배)는 {tqqq_tot * 100:+.1f}%, "
            f"QQQ(1배)는 {qqq_tot * 100:+.1f}%였습니다. "
            f"{summary_msg} 그동안 TQQQ의 최대낙폭은 {tqqq_mdd * 100:.1f}%였습니다."
        )
        st.info(msg)

        # 3-1 & 3-3. 누적 성과 곡선과 낙폭 그래프 나란히 배치
        use_log = st.checkbox("누적 성과 그래프 세로축 로그 스케일 적용", value=False)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("누적 성과 곡선 (시작점 = 100 정규화)")
            norm_lev = (lev_prices / lev_prices.iloc[0]) * 100.0
            fig_perf = px.line(
                norm_lev,
                x=norm_lev.index,
                y=lev_symbols,
                labels={"value": "정규화 가격 (100 기준)", "datetime": "거래일", "variable": "종목"},
                log_y=use_log,
            )
            fig_perf.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_right:
            st.subheader("고점 대비 낙폭 시계열 (Drawdown)")
            dd_df = pd.DataFrame(index=lev_prices.index)
            for sym in lev_symbols:
                dd_df[sym] = calc_drawdown_series(lev_prices[sym]) * 100.0

            fig_dd = px.line(
                dd_df,
                x=dd_df.index,
                y=lev_symbols,
                labels={"value": "고점 대비 낙폭 (%)", "datetime": "거래일", "variable": "종목"},
            )
            fig_dd.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # 3-2. 성과 요약표
        st.subheader("성과 요약 비교표")
        rows = []
        for sym in lev_symbols:
            t_ret = calc_total_return(lev_prices[sym])
            c_ret = calc_cagr(lev_prices[sym])
            vol = calc_volatility(lev_prices[sym])
            mdd = calc_mdd(lev_prices[sym])
            rows.append({
                "종목": sym,
                "총수익률": f"{t_ret * 100:+.1f}%",
                "연환산수익률(CAGR)": f"{c_ret * 100:.1f}%",
                "연변동성": f"{vol * 100:.1f}%",
                "최대낙폭(MDD)": f"{mdd * 100:.1f}%",
            })
        st.table(pd.DataFrame(rows))

        st.caption("총수익률: 구간 내 단순 누적 수익률 | 연환산수익률(CAGR): 기하평균 기준 연간 복리 성장률")
        st.caption("연변동성: 일간수익률 표준편차 × sqrt(252) | 최대낙폭(MDD): 직전 최고점 대비 가장 크게 하락했던 비율 (고점 대비 얼마나 빠졌는가)")

        # 3-4. 원금 회복 계산기
        st.subheader("원금 회복 계산기")
        drop_pct = st.slider("가상 계좌 하락률 (%)", min_value=0, max_value=90, value=50, step=5)
        needed_gain = (1.0 / (1.0 - (drop_pct / 100.0)) - 1.0) * 100.0 if drop_pct < 100 else 0.0

        st.metric(label="원금 회복에 필요한 상승률", value=f"{needed_gain:+.1f}%")
        st.markdown("-50% 하락은 +100% 상승이 있어야 원금입니다. 하락과 상승은 대칭이 아닙니다.")

        # 3-6. 시작일 변경 유도
        st.markdown("---")
        st.info("사이드바에서 분석 시작일을 2023-01-03으로 바꿔 보십시오. 결론이 뒤집힙니다.")

    # ==========================================
    # 탭 2 — 분산투자의 힘
    # ==========================================
    with tab2:
        st.header("자산 배분을 통한 위험 감소 및 분산투자 효과")

        daily_ret = sub_prices.pct_change().dropna()

        # 4-1. 상관계수 히트맵
        st.subheader("7종 ETF 일간수익률 상관계수 행렬")
        corr_matrix = daily_ret.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1.0,
            zmax=1.0,
            color_continuous_scale="RdBu_r",
            labels=dict(color="상관계수"),
        )
        fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        # 4-2. 히트맵 해석 문장
        spy_vti_c = corr_matrix.loc["SPY", "VTI"]
        qqq_tqqq_c = corr_matrix.loc["QQQ", "TQQQ"]
        gld_spy_c = corr_matrix.loc["GLD", "SPY"]

        st.markdown(f"""
        - **SPY와 VTI의 상관계수 ({spy_vti_c:.2f})**: 1에 매우 가깝습니다. 둘 다 보유하는 것은 실질적인 분산투자가 아닙니다.
        - **QQQ, QLD, TQQQ의 상관계수 ({qqq_tqqq_c:.2f})**: 일간 기준으로 완전히 동일하게 움직이며, 사실상 같은 자산군입니다.
        - **GLD와 타 주식형 종목 간의 낮은 상관계수 (SPY와 {gld_spy_c:.2f})**: 금 현물은 주식과 상관성이 매우 낮아 진정한 분산 효과를 제공합니다.
        """)

        # 4-3. 두 자산 배분 실험 (SPY & GLD)
        st.subheader("두 자산 배분 실험 (SPY & GLD)")
        st.caption("참고: 매일 정해진 비중으로 자산을 재조정(Daily Rebalancing)한다는 가정하에 계산됩니다.")

        spy_pct = st.slider("SPY(미국 대형주) 목표 비중 (%)", min_value=0, max_value=100, value=50, step=5)
        gld_pct = 100 - spy_pct

        w_spy = spy_pct / 100.0
        w_gld = gld_pct / 100.0

        p_series = calc_portfolio_series(sub_prices.index, daily_ret, w_spy, w_gld)

        p_tot = calc_total_return(p_series)
        p_vol = calc_volatility(p_series)
        p_mdd = calc_mdd(p_series)

        # SPY 100% 대조군
        spy_s = sub_prices["SPY"]
        spy_tot = calc_total_return(spy_s)
        spy_vol = calc_volatility(spy_s)
        spy_mdd = calc_mdd(spy_s)

        diff_tot = p_tot - spy_tot
        diff_vol = p_vol - spy_vol
        diff_mdd = p_mdd - spy_mdd

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label=f"포트폴리오 총수익률 (SPY {spy_pct}% / GLD {gld_pct}%)",
                value=f"{p_tot * 100:+.1f}%",
                delta=f"{diff_tot * 100:+.1f}%p (vs SPY 100%)",
            )
        with m2:
            st.metric(
                label="포트폴리오 연변동성",
                value=f"{p_vol * 100:.1f}%",
                delta=f"{diff_vol * 100:+.1f}%p (vs SPY 100%)",
                delta_color="inverse",
            )
        with m3:
            st.metric(
                label="포트폴리오 최대낙폭(MDD)",
                value=f"{p_mdd * 100:.1f}%",
                delta=f"{diff_mdd * 100:+.1f}%p (vs SPY 100%)",
                delta_color="off",
            )

        # 4-4. 비중별 위험-수익 산점도
        st.subheader("비중별 위험-수익 산점도 (SPY vs GLD)")
        scatter_rows = []
        for w in range(0, 101, 10):
            ws = w / 100.0
            wg = 1.0 - ws
            temp_p = calc_portfolio_series(sub_prices.index, daily_ret, ws, wg)
            t_ret = calc_total_return(temp_p)
            t_vol = calc_volatility(temp_p)
            scatter_rows.append({
                "weight_spy": w,
                "연변동성": t_vol * 100.0,
                "총수익률": t_ret * 100.0,
                "라벨": f"SPY {w}% / GLD {100 - w}%",
            })
        scatter_df = pd.DataFrame(scatter_rows)

        fig_sc = go.Figure()
        # 11개 비중 점
        fig_sc.add_trace(go.Scatter(
            x=scatter_df["연변동성"],
            y=scatter_df["총수익률"],
            mode="markers+text",
            text=scatter_df["weight_spy"].apply(lambda x: f"SPY {x}%"),
            textposition="top center",
            marker=dict(size=10, color="royalblue"),
            name="비중별 조합",
        ))
        # 현재 선택 지점 강조
        fig_sc.add_trace(go.Scatter(
            x=[p_vol * 100.0],
            y=[p_tot * 100.0],
            mode="markers",
            marker=dict(size=15, color="crimson", line=dict(width=2, color="black")),
            name=f"현재 선택 (SPY {spy_pct}%)",
        ))
        fig_sc.update_layout(
            xaxis_title="연변동성 (%)",
            yaxis_title="총수익률 (%)",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        min_vol_item = scatter_df.loc[scatter_df["연변동성"].idxmin()]
        gld_only_vol = scatter_df.iloc[0]["연변동성"]
        spy_only_vol = scatter_df.iloc[-1]["연변동성"]

        st.markdown(f"""
        **분산투자의 결론: 가장 안전한 포트폴리오는 섞은 쪽에 있습니다.**  
        이 그래프의 왼쪽 끝(최저 변동성 지점)은 **주식 100%({spy_only_vol:.1f}%)도 금 100%({gld_only_vol:.1f}%)도 아닌 SPY {min_vol_item['weight_spy']:.0f}% / GLD {100 - min_vol_item['weight_spy']:.0f}% 조합({min_vol_item['연변동성']:.1f}%)**입니다.  
        상관관계가 낮은 자산을 섞음으로써 개별 자산 각각보다 변동성을 낮출 수 있습니다.
        """)

        # 4-5. 자산군 성과 비교
        st.subheader("주요 자산군 누적 성과 비교 (SPY, GLD, SCHD)")
        asset_symbols = ["SPY", "GLD", "SCHD"]
        norm_assets = (sub_prices[asset_symbols] / sub_prices[asset_symbols].iloc[0]) * 100.0

        fig_assets = px.line(
            norm_assets,
            x=norm_assets.index,
            y=asset_symbols,
            labels={"value": "정규화 가격 (100 기준)", "datetime": "거래일", "variable": "자산군"},
        )
        fig_assets.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_assets, use_container_width=True)

        st.caption("주의: SCHD는 배당이 미포함된 종가 가격 기준 성과이므로 분배금을 포함한 실질 누적 수익률 대비 과소평가되어 있습니다.")


if __name__ == "__main__":
    main()
