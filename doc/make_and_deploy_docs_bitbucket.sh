#!/usr/bin/env bash
# build the docs
make clean
make html

git clone git@bitbucket.org:nist_microcal/nist_microcal.bitbucket.io.git
rm -rf nist_microcal.bitbucket.io/mass
cp -r _build/html nist_microcal.bitbucket.io/mass

# commit to master and push
cd nist_microcal.bitbucket.io
git add mass
git commit -m "update mass docs"
git push origin master
