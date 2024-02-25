# EGM Local Activation Time Detection

## Overview
This repository contains the implementation of a novel method for detecting local activation times (LAT) in electrograms (EGMs), particularly tailored for real-time analysis in atrial fibrillation scenarios. Our approach is rooted in the study published in Frontiers in Physiology, titled "Real-Time Rotational Activity Detection in Atrial Fibrillation."

## Methodology
The method centers on identifying the highest sustained negative slope in unipolar signals, a key marker in EGM LAT estimation. It employs a linear filter for signal processing, which is then adapted to various catheter topologies. Additionally, our system integrates a novel rotational activity detection algorithm, leveraging optical flow analysis and rotational pattern matching.

## Validation and Application
This system has been validated using a combination of in silico, experimental, and clinical signals. Its versatility allows for near real-time application, with adaptability to multiple catheter topologies, making it a significant tool for electrophysiologists studying atrial fibrillation.

## Usage
Instructions for setting up the environment, running the code, and performing analysis are provided in the respective folders.

## Contribution
We encourage contributions to this repository. Please read our contribution guidelines for more information.

## Citation
If you use this method in your research, please cite:
> Ríos-Muñoz GR, Arenal Á and Artés-Rodríguez A (2018) Real-Time Rotational Activity Detection in Atrial Fibrillation. Front. Physiol. 9:208. doi: 10.3389/fphys.2018.00208

## Funding

This project was partly supported by several grants and institutions, which have been instrumental in the development and validation of the EGM Local Activation Time Detection method. The funding sources include:

> MINECO/FEDER (ADVENTURE id. TEC2015-69868-C2-1-R)
> Comunidad de Madrid (project CASI-CAM-CM id. S2013/ICE-2845)
> Beca de la Sección de Electrofisiología y Arritmias de la SEC

We gratefully acknowledge the support of these organizations, which has been crucial in advancing this research.
