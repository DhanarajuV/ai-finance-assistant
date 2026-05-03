import plotly.graph_objects as go
import yfinance as yf


def portfolio_pie_chart(text: str) -> go.Figure:
    """Create a pie chart. Handles multiple formats:
    - 'AAPL 40%, GOOGL 30%'
    - '20000 in AAPL, 3000 in LMT'
    - 'AAPL: 46.5%'
    """
    import re
    names = []
    values = []

    # Try pattern: NAME: VALUE%
    matches = re.findall(r'[•\-\*]\s*(\w+):\s*([\d.]+)%', text)
    if matches:
        for name, val in matches:
            names.append(name)
            values.append(float(val))
    else:
        # Try pattern: NUMBER in NAME  or  NAME NUMBER
        for part in re.split(r',|and|\n', text):
            part = part.strip()
            if not part:
                continue

            # "20000 in AAPL" format
            m = re.match(r'([\d.]+)\s+in\s+(\w+)', part, re.IGNORECASE)
            if m:
                values.append(float(m.group(1)))
                names.append(m.group(2).upper())
                continue

            # "AAPL 40%" or "AAPL 40" format
            m = re.match(r'(\w+)\s+([\d.]+)%?', part)
            if m:
                names.append(m.group(1).upper())
                values.append(float(m.group(2)))

    if not names:
        return None

    fig = go.Figure(data=[go.Pie(labels=names, values=values, hole=0.4)])
    fig.update_layout(title="Portfolio Allocation", height=400)
    return fig


def stock_price_chart(symbol: str, period: str = "6mo") -> go.Figure:
    """Create a price history line chart for a stock."""
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#2196F3"),
        ))
        fig.update_layout(
            title=f"{symbol.upper()} - {period} Price History",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
        )
        return fig
    except Exception:
        return None

