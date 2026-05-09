"""Tests for the chart generation functions."""
from src.web_app.charts import portfolio_pie_chart, stock_price_chart


class TestPortfolioPieChart:
    """Test portfolio_pie_chart parsing and chart generation."""

    def test_basic_format(self):
        fig = portfolio_pie_chart("AAPL 40%, GOOGL 30%, BND 30%")
        assert fig is not None
        assert fig.data[0].labels == ("AAPL", "GOOGL", "BND")
        assert fig.data[0].values == (40.0, 30.0, 30.0)

    def test_bullet_format(self):
        text = "Holdings:\n  • AAPL: 50.0%\n  • GOOGL: 30.0%\n  • BND: 20.0%"
        fig = portfolio_pie_chart(text)
        assert fig is not None
        assert "AAPL" in fig.data[0].labels
        assert "GOOGL" in fig.data[0].labels
        assert "BND" in fig.data[0].labels

    def test_dash_bullet_format(self):
        text = "- AAPL: 60.0%\n- BND: 40.0%"
        fig = portfolio_pie_chart(text)
        assert fig is not None
        assert fig.data[0].labels == ("AAPL", "BND")

    def test_number_in_name_format(self):
        text = "20000 in AAPL, 10000 in BND"
        fig = portfolio_pie_chart(text)
        assert fig is not None
        assert "AAPL" in fig.data[0].labels
        assert "BND" in fig.data[0].labels

    def test_none_input(self):
        fig = portfolio_pie_chart(None)
        assert fig is None

    def test_empty_string(self):
        fig = portfolio_pie_chart("")
        assert fig is None

    def test_non_string_input(self):
        fig = portfolio_pie_chart(123)
        assert fig is None

    def test_list_input(self):
        fig = portfolio_pie_chart([{"type": "text", "text": "hello"}])
        assert fig is None

    def test_no_parseable_data(self):
        fig = portfolio_pie_chart("Hello, how are you today?")
        assert fig is None

    def test_donut_hole(self):
        fig = portfolio_pie_chart("AAPL 50%, BND 50%")
        assert fig is not None
        assert fig.data[0].hole == 0.4

    def test_and_separator(self):
        text = "5000 in AAPL and 3000 in BND"
        fig = portfolio_pie_chart(text)
        assert fig is not None
        assert "AAPL" in fig.data[0].labels
        assert "BND" in fig.data[0].labels


class TestStockPriceChart:
    """Test stock_price_chart generation."""

    def test_valid_symbol(self):
        fig = stock_price_chart("AAPL", "1mo")
        # May return None if market is closed or network issues
        if fig is not None:
            assert len(fig.data) == 1
            assert fig.data[0].name == "Close Price"

    def test_invalid_symbol(self):
        fig = stock_price_chart("ZZZZZZZ123")
        assert fig is None

    def test_default_period(self):
        fig = stock_price_chart("MSFT")
        # Default period is 6mo
        if fig is not None:
            assert "6mo" in fig.layout.title.text
