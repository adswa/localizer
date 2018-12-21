#!/bin/bash

#"This short script fixes an error Adina made in the structure of the
#dataset. The dataset lacked a session-directory for the avmovie session.
#This script creates the necessary directory and moves the directories
#that should live there inside."

set -e
set -u

subs=$(find -type d -maxdepth 1 -name 'sub-*' -printf "%f\n" | sort | uniq)
session=ses-movie
for sub in $subs; do
    sub_dir=$sub
    [ ! -d "${sub_dir}/${session}" ] && mkdir -p "${sub_dir}/${session}";
done

for sub in $subs; do
    sub_dir=$sub
    git mv {${sub_dir}/func,${sub}/anat,${sub}/xfm} ${sub_dir}/${session};
done
