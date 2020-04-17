#!/usr/bin/env bash
# build the docs
make clean
make html

git clone git@bitbucket.org:oneilg/oneilg.bitbucket.io.git
rm -rf oneilg.bitbucket.io/mass
mkdir oneilg.bitbucket.io/mass
cp -r _build/html oneilg.bitbucket.io/mass

# commit to master and push
cd oneilg.bitbucket.io
git add mass
git commit -m "update mass docs"
git push origin master
