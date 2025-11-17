FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..4] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    (* Layer 0 variables *)
    layer_0_output : ARRAY[0..4] OF REAL;

    (* Layer 1 variables *)
    weights_1 : ARRAY[0..4, 0..15] OF REAL;
    layer_1_output : ARRAY[0..15] OF REAL;

    (* Layer 2 variables *)
    bias_2 : ARRAY[0..15] OF REAL;
    layer_2_output : ARRAY[0..15] OF REAL;

    (* Layer 3 variables *)
    layer_3_output : ARRAY[0..15] OF REAL;

    (* Layer 4 variables *)
    weights_4 : ARRAY[0..15, 0..11] OF REAL;
    layer_4_output : ARRAY[0..11] OF REAL;

    (* Layer 5 variables *)
    bias_5 : ARRAY[0..11] OF REAL;
    layer_5_output : ARRAY[0..11] OF REAL;

    (* Layer 6 variables *)
    layer_6_output : ARRAY[0..11] OF REAL;

    (* Layer 7 variables *)
    weights_7 : ARRAY[0..11, 0..2] OF REAL;
    layer_7_output : ARRAY[0..2] OF REAL;

    (* Layer 8 variables *)
    bias_8 : ARRAY[0..2] OF REAL;
    layer_8_output : ARRAY[0..2] OF REAL;

    (* Layer 9 variables *)
    layer_9_output : ARRAY[0..2] OF REAL;

    (* Temporary computation variables *)
    i : INT;
    j : INT;
    sum : REAL;
    max_val : REAL;
    exp_sum : REAL;
    initialized : BOOL := FALSE;
END_VAR

(* Initialize weights (one-time setup) *)
IF NOT initialized THEN

    weights_1[0,0] := -0.095930;
    weights_1[0,1] := 0.000692;
    weights_1[0,2] := -0.604264;
    weights_1[0,3] := 0.007830;
    weights_1[0,4] := 0.027697;
    weights_1[0,5] := -0.110796;
    weights_1[0,6] := -0.121137;
    weights_1[0,7] := 0.119850;
    weights_1[0,8] := 0.057352;
    weights_1[0,9] := 0.002557;
    weights_1[0,10] := -0.385245;
    weights_1[0,11] := -0.190493;
    weights_1[0,12] := 0.015531;
    weights_1[0,13] := -0.107952;
    weights_1[0,14] := -0.004428;
    weights_1[0,15] := -0.455217;
    weights_1[1,0] := -0.228439;
    weights_1[1,1] := 0.000487;
    weights_1[1,2] := 0.502708;
    weights_1[1,3] := -0.011215;
    weights_1[1,4] := -0.003850;
    weights_1[1,5] := 0.117060;
    weights_1[1,6] := 0.120599;
    weights_1[1,7] := -0.019681;
    weights_1[1,8] := -0.033640;
    weights_1[1,9] := -0.174245;
    weights_1[1,10] := 0.352214;
    weights_1[1,11] := 0.226439;
    weights_1[1,12] := -0.073395;
    weights_1[1,13] := 0.115894;
    weights_1[1,14] := 0.001815;
    weights_1[1,15] := 0.047289;
    weights_1[2,0] := 0.130232;
    weights_1[2,1] := 0.009477;
    weights_1[2,2] := -0.059794;
    weights_1[2,3] := -0.001728;
    weights_1[2,4] := -0.190443;
    weights_1[2,5] := 0.033723;
    weights_1[2,6] := -0.017326;
    weights_1[2,7] := -0.050876;
    weights_1[2,8] := -0.119164;
    weights_1[2,9] := -0.004911;
    weights_1[2,10] := 0.035715;
    weights_1[2,11] := -0.020005;
    weights_1[2,12] := 0.085858;
    weights_1[2,13] := -0.066041;
    weights_1[2,14] := -0.001310;
    weights_1[2,15] := -0.063746;
    weights_1[3,0] := -0.051169;
    weights_1[3,1] := 0.001477;
    weights_1[3,2] := 0.372914;
    weights_1[3,3] := 0.067524;
    weights_1[3,4] := 0.337182;
    weights_1[3,5] := 0.028778;
    weights_1[3,6] := 0.043451;
    weights_1[3,7] := -0.477387;
    weights_1[3,8] := 0.013193;
    weights_1[3,9] := -0.300537;
    weights_1[3,10] := 0.073491;
    weights_1[3,11] := 0.131580;
    weights_1[3,12] := -0.126345;
    weights_1[3,13] := 0.092612;
    weights_1[3,14] := 0.000971;
    weights_1[3,15] := 0.090229;
    weights_1[4,0] := -0.105823;
    weights_1[4,1] := 0.247740;
    weights_1[4,2] := -0.379393;
    weights_1[4,3] := 0.370569;
    weights_1[4,4] := -0.515837;
    weights_1[4,5] := -0.246769;
    weights_1[4,6] := 0.246283;
    weights_1[4,7] := -0.544111;
    weights_1[4,8] := -0.117840;
    weights_1[4,9] := -0.010282;
    weights_1[4,10] := -0.288469;
    weights_1[4,11] := -0.503179;
    weights_1[4,12] := -0.058943;
    weights_1[4,13] := -0.247450;
    weights_1[4,14] := -0.083657;
    weights_1[4,15] := -0.015038;

    bias_2[0] := -0.980481;
    bias_2[1] := -2.405819;
    bias_2[2] := 0.690016;
    bias_2[3] := -2.993280;
    bias_2[4] := 0.040576;
    bias_2[5] := 1.595755;
    bias_2[6] := -2.240234;
    bias_2[7] := 1.722846;
    bias_2[8] := 0.410530;
    bias_2[9] := -0.547844;
    bias_2[10] := 2.407200;
    bias_2[11] := 0.369404;
    bias_2[12] := 2.175193;
    bias_2[13] := 2.996372;
    bias_2[14] := 2.551130;
    bias_2[15] := -0.598656;


    weights_4[0,0] := -0.261270;
    weights_4[0,1] := 0.107119;
    weights_4[0,2] := -0.057380;
    weights_4[0,3] := 0.239176;
    weights_4[0,4] := -0.531469;
    weights_4[0,5] := 0.012263;
    weights_4[0,6] := 0.035657;
    weights_4[0,7] := -0.096014;
    weights_4[0,8] := 0.004310;
    weights_4[0,9] := 0.120793;
    weights_4[0,10] := -0.087786;
    weights_4[0,11] := 0.030782;
    weights_4[1,0] := -0.055516;
    weights_4[1,1] := 0.199512;
    weights_4[1,2] := -0.682147;
    weights_4[1,3] := -0.007604;
    weights_4[1,4] := -0.110083;
    weights_4[1,5] := -0.045768;
    weights_4[1,6] := -0.092431;
    weights_4[1,7] := -1.149229;
    weights_4[1,8] := 0.249240;
    weights_4[1,9] := 0.167818;
    weights_4[1,10] := -1.157714;
    weights_4[1,11] := 0.341394;
    weights_4[2,0] := 0.135732;
    weights_4[2,1] := -0.008291;
    weights_4[2,2] := 0.049353;
    weights_4[2,3] := -0.514215;
    weights_4[2,4] := -0.152022;
    weights_4[2,5] := -0.251635;
    weights_4[2,6] := -0.153912;
    weights_4[2,7] := 0.006585;
    weights_4[2,8] := 0.149169;
    weights_4[2,9] := -0.086582;
    weights_4[2,10] := -0.029337;
    weights_4[2,11] := 0.035312;
    weights_4[3,0] := -0.020302;
    weights_4[3,1] := -0.011110;
    weights_4[3,2] := -0.289731;
    weights_4[3,3] := -0.008050;
    weights_4[3,4] := 0.191495;
    weights_4[3,5] := -0.007147;
    weights_4[3,6] := 0.219508;
    weights_4[3,7] := -0.448034;
    weights_4[3,8] := 0.133700;
    weights_4[3,9] := -0.208194;
    weights_4[3,10] := -0.342866;
    weights_4[3,11] := 0.048072;
    weights_4[4,0] := -0.335608;
    weights_4[4,1] := -0.274515;
    weights_4[4,2] := -0.048835;
    weights_4[4,3] := -0.236680;
    weights_4[4,4] := -0.090380;
    weights_4[4,5] := 0.036612;
    weights_4[4,6] := -0.205669;
    weights_4[4,7] := 0.064767;
    weights_4[4,8] := -0.502523;
    weights_4[4,9] := -0.562775;
    weights_4[4,10] := 0.037710;
    weights_4[4,11] := -0.180757;
    weights_4[5,0] := -0.442252;
    weights_4[5,1] := -0.208702;
    weights_4[5,2] := 0.103663;
    weights_4[5,3] := -0.104223;
    weights_4[5,4] := -0.398669;
    weights_4[5,5] := 0.493328;
    weights_4[5,6] := -0.170170;
    weights_4[5,7] := 0.077624;
    weights_4[5,8] := -0.503384;
    weights_4[5,9] := -0.263422;
    weights_4[5,10] := 0.144634;
    weights_4[5,11] := -0.085226;
    weights_4[6,0] := 0.094759;
    weights_4[6,1] := 0.184407;
    weights_4[6,2] := -0.651972;
    weights_4[6,3] := -0.008861;
    weights_4[6,4] := -0.043380;
    weights_4[6,5] := 0.275419;
    weights_4[6,6] := -0.045682;
    weights_4[6,7] := -0.842328;
    weights_4[6,8] := -0.076355;
    weights_4[6,9] := 0.106007;
    weights_4[6,10] := -0.005826;
    weights_4[6,11] := -0.201158;
    weights_4[7,0] := -0.556553;
    weights_4[7,1] := -0.327094;
    weights_4[7,2] := 0.002942;
    weights_4[7,3] := -0.452641;
    weights_4[7,4] := -0.159332;
    weights_4[7,5] := 0.310186;
    weights_4[7,6] := -0.189237;
    weights_4[7,7] := 0.007894;
    weights_4[7,8] := -0.294373;
    weights_4[7,9] := -0.186090;
    weights_4[7,10] := 0.014068;
    weights_4[7,11] := -0.326161;
    weights_4[8,0] := -0.345027;
    weights_4[8,1] := 0.073290;
    weights_4[8,2] := 0.050642;
    weights_4[8,3] := -0.107399;
    weights_4[8,4] := -0.324269;
    weights_4[8,5] := 0.381255;
    weights_4[8,6] := -0.053493;
    weights_4[8,7] := -0.067163;
    weights_4[8,8] := -0.129507;
    weights_4[8,9] := -0.023505;
    weights_4[8,10] := 0.082253;
    weights_4[8,11] := -0.175474;
    weights_4[9,0] := 0.188168;
    weights_4[9,1] := 0.063347;
    weights_4[9,2] := 0.019446;
    weights_4[9,3] := -0.180079;
    weights_4[9,4] := -0.276512;
    weights_4[9,5] := 0.228827;
    weights_4[9,6] := -0.089237;
    weights_4[9,7] := -0.002963;
    weights_4[9,8] := -0.043670;
    weights_4[9,9] := -0.109243;
    weights_4[9,10] := -0.101433;
    weights_4[9,11] := -0.100929;
    weights_4[10,0] := -0.011021;
    weights_4[10,1] := 0.048698;
    weights_4[10,2] := 0.110389;
    weights_4[10,3] := -1.080399;
    weights_4[10,4] := -0.104138;
    weights_4[10,5] := -0.112609;
    weights_4[10,6] := -0.363702;
    weights_4[10,7] := 0.054155;
    weights_4[10,8] := -0.059634;
    weights_4[10,9] := -0.578229;
    weights_4[10,10] := 0.102744;
    weights_4[10,11] := -0.036446;
    weights_4[11,0] := -0.331553;
    weights_4[11,1] := -0.071238;
    weights_4[11,2] := -0.032371;
    weights_4[11,3] := -0.388269;
    weights_4[11,4] := -0.216394;
    weights_4[11,5] := 0.064683;
    weights_4[11,6] := -0.123483;
    weights_4[11,7] := 0.063245;
    weights_4[11,8] := -0.495136;
    weights_4[11,9] := -0.160000;
    weights_4[11,10] := 0.251014;
    weights_4[11,11] := -0.257789;
    weights_4[12,0] := -0.521276;
    weights_4[12,1] := -0.339843;
    weights_4[12,2] := 0.102932;
    weights_4[12,3] := -0.848057;
    weights_4[12,4] := -0.042467;
    weights_4[12,5] := 0.023446;
    weights_4[12,6] := -0.108048;
    weights_4[12,7] := 0.087676;
    weights_4[12,8] := 0.072063;
    weights_4[12,9] := -0.154295;
    weights_4[12,10] := 0.227691;
    weights_4[12,11] := -0.163715;
    weights_4[13,0] := -0.975383;
    weights_4[13,1] := -0.318735;
    weights_4[13,2] := 0.041320;
    weights_4[13,3] := -1.233760;
    weights_4[13,4] := -0.324155;
    weights_4[13,5] := 0.115972;
    weights_4[13,6] := -0.496875;
    weights_4[13,7] := 0.013090;
    weights_4[13,8] := -0.216962;
    weights_4[13,9] := -0.313846;
    weights_4[13,10] := 0.286013;
    weights_4[13,11] := -0.487865;
    weights_4[14,0] := -1.711578;
    weights_4[14,1] := -1.448045;
    weights_4[14,2] := 0.074641;
    weights_4[14,3] := -0.485800;
    weights_4[14,4] := -0.881306;
    weights_4[14,5] := -0.269586;
    weights_4[14,6] := -0.601741;
    weights_4[14,7] := -0.125377;
    weights_4[14,8] := -1.803676;
    weights_4[14,9] := -1.520676;
    weights_4[14,10] := 0.261292;
    weights_4[14,11] := -1.616396;
    weights_4[15,0] := 0.171718;
    weights_4[15,1] := 0.335709;
    weights_4[15,2] := -0.022494;
    weights_4[15,3] := 0.007286;
    weights_4[15,4] := 0.136407;
    weights_4[15,5] := 0.098455;
    weights_4[15,6] := 0.311768;
    weights_4[15,7] := 0.027948;
    weights_4[15,8] := 0.296725;
    weights_4[15,9] := -0.252980;
    weights_4[15,10] := -0.009506;
    weights_4[15,11] := 0.367563;

    bias_5[0] := 1.481165;
    bias_5[1] := -0.758992;
    bias_5[2] := 1.639378;
    bias_5[3] := 1.376125;
    bias_5[4] := -0.883914;
    bias_5[5] := -1.043172;
    bias_5[6] := -1.322489;
    bias_5[7] := 1.710258;
    bias_5[8] := -0.570090;
    bias_5[9] := -0.204427;
    bias_5[10] := 1.442890;
    bias_5[11] := -0.280403;


    weights_7[0,0] := -0.542195;
    weights_7[0,1] := -0.230593;
    weights_7[0,2] := 0.187449;
    weights_7[1,0] := 0.082667;
    weights_7[1,1] := -0.085455;
    weights_7[1,2] := 0.225467;
    weights_7[2,0] := 0.634390;
    weights_7[2,1] := -0.168816;
    weights_7[2,2] := 0.196424;
    weights_7[3,0] := -1.856785;
    weights_7[3,1] := 0.146151;
    weights_7[3,2] := 0.590299;
    weights_7[4,0] := -0.091240;
    weights_7[4,1] := -0.414188;
    weights_7[4,2] := 0.139226;
    weights_7[5,0] := -0.118131;
    weights_7[5,1] := -0.651260;
    weights_7[5,2] := -0.143628;
    weights_7[6,0] := 0.271132;
    weights_7[6,1] := -0.104659;
    weights_7[6,2] := 0.444165;
    weights_7[7,0] := 0.151004;
    weights_7[7,1] := -0.906230;
    weights_7[7,2] := -0.211486;
    weights_7[8,0] := -0.399137;
    weights_7[8,1] := -0.514309;
    weights_7[8,2] := -0.257237;
    weights_7[9,0] := 0.047009;
    weights_7[9,1] := -0.300323;
    weights_7[9,2] := 0.339689;
    weights_7[10,0] := 0.031179;
    weights_7[10,1] := -0.438313;
    weights_7[10,2] := -0.167528;
    weights_7[11,0] := 0.223605;
    weights_7[11,1] := 0.039704;
    weights_7[11,2] := 0.393325;

    bias_8[0] := -0.692060;
    bias_8[1] := 1.614790;
    bias_8[2] := -1.157133;


    initialized := TRUE;
END_IF;

(* Forward pass computation *)
(* Layer 0: Reshape (copy input to output) *)
FOR i := 0 TO 4 DO
    layer_0_output[i] := input_data[i];
END_FOR;

(* Layer 1: MatMul *)
FOR j := 0 TO 15 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + layer_0_output[i] * weights_1[i,j];
    END_FOR;
    layer_1_output[j] := sum;
END_FOR;

(* Layer 2: Add (Bias) *)
FOR i := 0 TO 15 DO
    layer_2_output[i] := layer_1_output[i] + bias_2[i];
END_FOR;

(* Layer 3: Activation (RELU) *)
FOR i := 0 TO 15 DO
    layer_3_output[i] := MAX(layer_2_output[i], 0.0);
END_FOR;

(* Layer 4: MatMul *)
FOR j := 0 TO 11 DO
    sum := 0.0;
    FOR i := 0 TO 15 DO
        sum := sum + layer_3_output[i] * weights_4[i,j];
    END_FOR;
    layer_4_output[j] := sum;
END_FOR;

(* Layer 5: Add (Bias) *)
FOR i := 0 TO 11 DO
    layer_5_output[i] := layer_4_output[i] + bias_5[i];
END_FOR;

(* Layer 6: Activation (RELU) *)
FOR i := 0 TO 11 DO
    layer_6_output[i] := MAX(layer_5_output[i], 0.0);
END_FOR;

(* Layer 7: MatMul *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 11 DO
        sum := sum + layer_6_output[i] * weights_7[i,j];
    END_FOR;
    layer_7_output[j] := sum;
END_FOR;

(* Layer 8: Add (Bias) *)
FOR i := 0 TO 2 DO
    layer_8_output[i] := layer_7_output[i] + bias_8[i];
END_FOR;

(* Layer 9: Activation (SOFTMAX) *)
max_val := layer_8_output[0];
FOR i := 1 TO 2 DO
    IF layer_8_output[i] > max_val THEN
        max_val := layer_8_output[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(layer_8_output[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK