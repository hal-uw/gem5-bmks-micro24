#!/bin/bash
while read j; do
    cp $j/* ../$j/
done < file_list
