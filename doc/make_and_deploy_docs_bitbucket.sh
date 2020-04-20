#!/usr/bin/env bash
# build the docs
make clean
make html


git clone git@bitbucket.org:oneilg/oneilg.bitbucket.io.git

if [[ $1 = master ]]
then
echo "deply for master"
rm -rf oneilg.bitbucket.io/mass
cp -r _build/html oneilg.bitbucket.io/mass
else
echo "deploy for non master"
rm -rf oneilg.bitbucket.io/mass_non_master
cp -r _build/html oneilg.bitbucket.io/mass_non_master
fi

# commit to master and push
cd oneilg.bitbucket.io
git add mass
git commit -m "update mass docs"
git push origin master
