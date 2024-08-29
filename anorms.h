// anorms.h -- Vector Norms (not Normalization) -- Aussie AI Base Library  
// Created 12th Nov 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YNORMS_INCLUDE_HEADER_H
#define AUSSIE_YNORMS_INCLUDE_HEADER_H


//---------------------------------------------------
// VECTOR NORMS
//---------------------------------------------------

float aussie_vector_L1_norm(float v[], int n);
float aussie_vector_L2_norm(float v[], int n);
float aussie_vector_L2_squared_norm(float v[], int n);
float aussie_vector_L3_norm(float v[], int n);
float aussie_vector_L1_norm_if_test(float v[], int n);
float aussie_vector_L1_norm_bitwise_fabs(float v[], int n);

//---------------------------------------------------
//---------------------------------------------------

void aussie_yvector_norm_unit_tests();  // Unit Test NORMs

//---------------------------------------------------
//---------------------------------------------------


#endif //AUSSIE_YNORMS_INCLUDE_HEADER_H

