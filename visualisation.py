import plotly.graph_objects as go

def create_candlestick_chart(df, symbol, show_mas=True, interval='1d'):
    """
    Create an interactive candlestick chart with technical indicators
    
    Args:
        df: DataFrame with OHLC data and moving averages
        symbol: Stock ticker symbol
        show_mas: Whether to display moving averages
        interval: Data interval for appropriate range buttons
    
    Returns:
        Plotly figure object
    """
    # Create Candlestick Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol,
        increasing_line_color='green',
        decreasing_line_color='red',
        line=dict(width=1.5)  # Make candlesticks more visible
    ))
    
    # Add Moving Averages if requested
    if show_mas:
        ma_colors = {
            'MA10': '#1f77b4',  # Blue
            'MA20': '#ff7f0e',  # Orange
            'MA60': '#2ca02c',  # Green
            'MA200': '#d62728'  # Red
        }
        
        for ma, color in ma_colors.items():
            if ma in df.columns and not df[ma].isnull().all():
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color, width=1.5),
                    connectgaps=True
                ))
    
    # Add volume as a bar chart at the bottom
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(color='rgba(100, 100, 100, 0.5)'),
        yaxis='y2'
    ))
    
    # Determine suitable range buttons based on selected timeframe
    if interval in ['5m', '15m', '30m']:
        range_buttons = [
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=3, label="3d", step="day", stepmode="backward"),
            dict(count=5, label="5d", step="day", stepmode="backward"),
            dict(step="all")
        ]
    elif interval == '1h':
        range_buttons = [
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=5, label="5d", step="day", stepmode="backward"),
            dict(count=10, label="10d", step="day", stepmode="backward"),
            dict(step="all")
        ]
    else:
        range_buttons = [
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    
    # Enhance the layout
    fig.update_layout(
        title={
            'text': f"Technical Analysis for {symbol}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=range_buttons
            ),
            showline=True, 
            linewidth=1.2, 
            linecolor='black',
            type='date'  # Ensure proper date handling
        ),
        yaxis=dict(
            fixedrange=False,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            title="Price"
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            visible=False
        ),
        autosize=True,
        height=600,
        hovermode="x unified",
        plot_bgcolor="#f5f5f5",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig