cd overleaf
git pull
cp ../experiments/**/*.png fig
cp ../additional_figures/*.png fig
git add fig/*.png
git commit -m "updated figures"
git fetch
git push
