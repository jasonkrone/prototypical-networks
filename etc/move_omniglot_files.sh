#!/bin/bash

# background images
for f in $(find omniglot/python/images_background/ -name "*.png"); do
    fname="${f##*/}"
    echo "$fname"
    example="${fname#*_}"
    echo "$class"
    class="${fname%_*}"
    echo "$example"
    mkdir -p omniglot_images/"$class"
    mv "$f" omniglot_images/"$class"/"$example"
done

# evaluation images
for f in $(find omniglot/python/images_evaluation/ -name "*.png"); do
    fname="${f##*/}"
    echo "$fname"
    example="${fname#*_}"
    echo "$class"
    class="${fname%_*}"
    echo "$example"
    mkdir -p omniglot_images/"$class"
    mv "$f" omniglot_images/"$class"/"$example"
done


