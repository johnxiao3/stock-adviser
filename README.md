# Stock-Adviser

## Project Overview
**Stock-Adviser** is an image gallery-based stock screener designed to analyze stock data daily and provide potential investment recommendations. Using the last six months of daily stock data, it applies indicators such as MACD, DIF, MACD histogram, and EMA lines to create easy-to-view visualizations that help identify promising stocks from a large selection of over 3,600 stocks. Dynamic image loading in the gallery enables smooth browsing through stock images without overwhelming system resources.

## Features
- **Daily Stock Screening**: Analyzes the last 6 months of daily stock data to identify investment opportunities.
- **MACD and EMA Indicators**: Plots MACD, DIF, MACD histogram, and EMA lines for each stock, providing comprehensive indicator information.
- **Dynamic Image Loading**: Initially loads 10 images and loads additional images as the user scrolls for better performance.
- **Summary Statistics**: Displays the number of selected stocks from the dataset of 3,600 stocks.

## Planned Features
- Display the image name in the gallery title with the format `image_name (index/total)`.
- Sort images by filename and date to enhance navigation.
- Introduce advanced sorting options and add more financial indicators.

## Setup and Installation
### Prerequisites
- **Docker**: Make sure Docker is installed on your system.

### Building the Project
1. **Create a Docker Volume**:
   ```bash
   docker volume create image_gallery_volume
2. **Build the Docker image:
   ```bash
   docker build -t stock-adviser .
3. **Run the Docker Container:
   ```bash
   docker run -d --name stock-adviser-container -p 80:5000 -v image_gallery_volume:/app/static stock-adviser
