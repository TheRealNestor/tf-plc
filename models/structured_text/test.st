PROGRAM HelloWorld
  VAR
    OutputReg AT %MD100 : REAL;
    Counter AT %MD104 : REAL;
  END_VAR

  Counter := Counter + 1.0;
  OutputReg := Counter;
END_PROGRAM


CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms,PRIORITY := 0);
    PROGRAM instance0 WITH task0 : HelloWorld;
  END_RESOURCE
END_CONFIGURATION