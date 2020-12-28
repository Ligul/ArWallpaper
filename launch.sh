#!/usr/bin/env bash

pgrep devilspie2
if [ $? -eq 0 ]
then
  pkill devilspie2
fi

pgrep ArWallpaper
if [ $? -eq 0 ]
then
  pkill ArWallpaper
fi

/opt/ArWallpaper/ArWallpaper | \
while IFS= read -r line
do
  if [[ $line == *"Loaded texture:"* ]]
  then
		echo "=============================="
  	echo "   Window created, resizing"
  	echo "=============================="
  	sleep 0.1
  	devilspie2 --debug -f ./
  fi
done
