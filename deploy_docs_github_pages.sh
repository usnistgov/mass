#!/usr/bin/env bash
# build the docs
cd doc
make clean
make html
cd ..
# commit and push
git add -A
git commit -m "building and pushing docs"
git push github docs
# switch branches and pull the data we want
git checkout gh-pages
rm -rf .
touch .nojekyll
git checkout docs docs/build/html
mv ./docs/build/html/* ./
rm -rf ./docs
git add -A
git commit -m "publishing updated docs..."
git push github gh-pages
# switch back
git checkout docs
