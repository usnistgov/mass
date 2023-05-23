#!/usr/bin/env bash
# deploy the docs

git clone git@bitbucket.org:oneilg/oneilg.bitbucket.io.git

if [[ $1 = master ]]
then
echo "deploy for master"
rm -rf oneilg.bitbucket.io/mass
cp -r _build/html oneilg.bitbucket.io/mass
cd oneilg.bitbucket.io
git add mass
else
echo "deploy for non-master"
rm -rf oneilg.bitbucket.io/mass_non_master
cp -r _build/html oneilg.bitbucket.io/mass_non_master
cd oneilg.bitbucket.io
git add mass_non_master
fi

# commit to master and push
git commit -m "update mass docs"
git push origin master
