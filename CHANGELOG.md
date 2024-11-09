# Change Log
## [1.0.0] - 2024-11-7
### Feature added:
- sort the png file 
- add the docker-compose.yml file and update the readme
- add the redeploy.sh everytime I just need to run the ./redeploy.sh it will automatically pull the code and deploy it. That is great.
- the date used reverse sort now, it will show the recent dates first.
- added the .gitignore file and ignore the static file, those file should be created by another script rather can store in the code.
- fix the time calculation bug if the system is start the same time it will skip one day.
- add code to find buy and sell point for all the filtered stocks.
- add the buy stock selection based on the nearest buy result to be 0
- add the daily check based on the stock holding saved in the static/investment_plan.json.
- add feature daily check result saved in the static/log.txt. and one more page add to display the log.txt.
- add a download feature to download the filtered stocks.
- add button to jump to the first stock in the slider view.


## [1.0.0] - 2024-10-25
### Added
- Initial project setup with dynamic loading image gallery.
- Daily stock screener using MACD, DIF, and EMA lines.
- Summary statistic of selected stocks.

## [Upcoming]
### Planned Features
- Automatically udpate the stock result everyday
- Display image name and index in title.
- Sort images by file name and date.
- Improve sorting functionality and add advanced indicators.
