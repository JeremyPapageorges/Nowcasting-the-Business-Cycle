import matplotlib.pyplot as plt

def graph(series,title_1, label_1, y_label_1, title_2, label_2, y_label_2):

    fig, axs = plt.subplots(2,1, figsize = (10,6))

    kkr_purple = '#590e5b'  # Approximate hex color from the KKR logo
    marker_color = '#FF69B4'  # Another shade for the markers (pink)

    # First subplot: Full year-over-year percentage change
    axs[0].plot(series, color=kkr_purple, label= label_1)
    axs[0].set_title(title_1)
    axs[0].set_ylabel(y_label_1)
    axs[0].legend(loc="upper left")  # Add legend here with label
    axs[0].grid(True)
    # Second subplot: Last 12 months with mean line and markers
    axs[1].plot(series.iloc[-12:], color=kkr_purple, label= label_2)
    axs[1].scatter(series.iloc[-12:].index, series.iloc[-12:], color=marker_color, label="Monthly Data")
    axs[1].axhline(series.mean(), color='red', linestyle='--', linewidth=1, label="Mean")
    axs[1].set_title(title_2)
    axs[1].set_ylabel(y_label_2)
    axs[1].set_xlabel("Date")
    axs[1].legend(loc="upper left")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show