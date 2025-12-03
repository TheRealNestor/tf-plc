FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..4] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR CONSTANT
    weights_9 : ARRAY[0..49] OF SINT := [-81, 123, -74, -49, 41, -68, -44, 127, -13, 12, 92, -23, -115, 27, -112, -3, -114, 84, 62, 58, 58, 127, -107, -60, -39, -35, 127, -34, -127, -116, -127, 102, -127, 97, -127, 127, -60, 39, 23, -86, -112, 28, 65, 127, 113, -8, 92, -16, 53, 127];
    weight_scale_9 : ARRAY[0..9] OF REAL := [0.005487, 0.005121, 0.004429, 0.006029, 0.003847, 0.004914, 0.003670, 0.004026, 0.003241, 0.003767];
    weight_zero_point_9 : SINT := 0;
    bias_9 : ARRAY[0..9] OF REAL := [-0.077728, -0.486146, -0.055262, -2.552989, 0.013622, 2.175047, 2.147089, 1.386354, -0.004372, 1.996328];

    weights_14 : ARRAY[0..99] OF SINT := [44, -47, -7, 12, 56, 57, 77, -54, -43, -76, -46, 49, 15, -40, -54, 78, 34, 46, -91, -29, -52, -4, -37, 127, 124, -62, -5, -20, 48, 24, 96, 120, 57, -88, 31, -36, -80, -69, 127, 54, 87, 28, 25, -38, 94, 51, 127, -60, 117, -46, 57, -127, 116, -119, -127, -55, 59, 106, -52, 47, 127, -55, -127, 82, 3, -127, -105, 99, -82, 2, -65, -29, 86, -100, -94, 13, -89, 22, 12, 86, -22, 20, -12, -43, -62, -26, -42, 21, 29, -127, -25, -119, 100, -55, 30, 41, 5, 127, 72, 38];
    weight_scale_14 : ARRAY[0..9] OF REAL := [0.006375, 0.010391, 0.004515, 0.004288, 0.003916, 0.006372, 0.003891, 0.007257, 0.004664, 0.004674];
    weight_zero_point_14 : SINT := 0;
    bias_14 : ARRAY[0..9] OF REAL := [-0.366862, -1.956535, 0.023151, 0.000000, -0.049086, -0.836756, 0.000000, 2.007301, 0.051818, 0.161092];

    weights_19 : ARRAY[0..29] OF SINT := [-7, -16, 127, -127, -127, -23, -18, 26, -30, 31, 34, 12, -69, 69, -69, -10, -42, -24, -1, 23, 4, 94, 125, -14, -8, -30, 26, 33, -93, -30];
    weight_scale_19 : ARRAY[0..2] OF REAL := [0.011622, 0.005713, 0.006394];
    weight_zero_point_19 : SINT := 0;
    bias_19 : ARRAY[0..2] OF REAL := [1.010356, 0.425502, -1.451074];

END_VAR

VAR
    (* Layer 9 output variable *)
    layer_9_output : ARRAY[0..9] OF REAL;

    (* Layer 14 output variable *)
    layer_14_output : ARRAY[0..9] OF REAL;

    (* Temporary computation variables *)
    i : DINT;
    j : DINT;
    sum : REAL;
    max_val : REAL;
    exp_sum : REAL;

END_VAR

(* Layer 9: Fused Linear + RELU *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + input_data[i] * (weight_scale_9[j] * SINT_TO_REAL(weights_9[i * 10 + j] - weight_zero_point_9));
    END_FOR;
    layer_9_output[j] := MAX(sum + bias_9[j], 0.0);
END_FOR;

(* Layer 14: Fused Linear + RELU *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_9_output[i] * (weight_scale_14[j] * SINT_TO_REAL(weights_14[i * 10 + j] - weight_zero_point_14));
    END_FOR;
    layer_14_output[j] := MAX(sum + bias_14[j], 0.0);
END_FOR;

(* Layer 19: Fused Linear + SOFTMAX *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_14_output[i] * (weight_scale_19[j] * SINT_TO_REAL(weights_19[i * 3 + j] - weight_zero_point_19));
    END_FOR;
    output_data[j] := sum + bias_19[j];
END_FOR;

max_val := output_data[0];
FOR i := 1 TO 2 DO
    IF output_data[i] > max_val THEN
        max_val := output_data[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(output_data[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK
