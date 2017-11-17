#!/usr/bin/env bash
# build the docs
cd doc
make clean
make html
cd ..


# delete all in oneilg.bitbucket.io
rm -rf ../oneilg.bitbucket.io/mass
# copy html docs
mkdir ../oneilg.bitbucket.io/mass
cp -r doc/_build/html/* ../oneilg.bitbucket.io/mass

# commit to master
cd ../oneilg.bitbucket.io/mass
git add .
git commit -m "update mass docs"
git push origin master

# return to start directory
cd ../mass
