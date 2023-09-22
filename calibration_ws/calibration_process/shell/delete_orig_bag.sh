#!/bin/bash

delete_origin()
{
    local file_path=$1
    cd  $file_path
    rm */*.orig.bag
}

delete_origin ~/data/bags
delete_origin ~/data/ssd
delete_origin ~/data/sda_1
