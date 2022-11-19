#!/usr/bin/env zsh
cd overleaf || exit
git pull
cp ../experiments/**/*.png fig
cp ../additional_figures/*.png fig
git add fig/*.png
cp ../experiments/**/*.eps fig
cp ../additional_figures/*.eps fig
git add fig/*.eps
git commit -m "updated figures"
git fetch
git push
