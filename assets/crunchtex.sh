#! /bin/bash
magick \( ${1}_Color.png -set colorspace sRGB -resize $2 -depth 8 \) \
    \( ${1}_AmbientOcclusion.png -set colorspace LinearGray -resize $2 -depth 8 \) \
    -channel-fx '| gray=>alpha' ${3}.co.png
magick \( ${1}_NormalGL.png -set colorspace RGB -resize $2 -depth 8 \) \
    \( ${1}_Roughness.png -set colorspace LinearGray -resize $2 -depth 8 \) \
    -channel-fx '| gray=>alpha' ${3}.nr.png