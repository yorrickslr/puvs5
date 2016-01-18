# puvs5
Parallele und verteilte Systeme - Übung 5


Das verwendete System für Aufgabe 2 hat folgende Spezifikationen:
	CPU:	AMD Athlon II x2 240
	RAM:	8Gb DDR3 @1333MHz
	GraKa:	Palit GTX 560 OC
	MoBo:	Asrock N68-VS3 FX
	SSD:	OCZ Vector 150

Gerade bei der Berechnung einer 10'000 x 10'000 x 10'000 großen Matrix ist der Speedup beängstigend:
Die Grafikkarte braucht nur wenige Sekunden, während die CPU noch 50 Minuten später rechnet.
Danach haben wir das Programm abgebrochen.

C:\Users\Admin\Documents\Visual Studio 2015\Projects\opencl\x64\Release>opencl.exe 1000 1000 1000
Matrix sizes C[1000][1000] = A[1000][1000] x B[1000][1000]
Running serial algorithm...
Checking results... ok.
Showing stats...
   serial runtime = 12.022929
   OpenCL runtime = 0.088727
   Speedup = 135.505127

C:\Users\Admin\Documents\Visual Studio 2015\Projects\opencl\x64\Release>opencl.exe 1111 1111 1111
Matrix sizes C[1111][1111] = A[1111][1111] x B[1111][1111]
Running serial algorithm...
Checking results... ok.
Showing stats...
   serial runtime = 18.129139
   OpenCL runtime = 0.138511
   Speedup = 130.885956

C:\Users\Admin\Documents\Visual Studio 2015\Projects\opencl\x64\Release>opencl.exe 100 100 100
Matrix sizes C[100][100] = A[100][100] x B[100][100]
Running serial algorithm...
Checking results... ok.
Showing stats...
   serial runtime = 0.001641
   OpenCL runtime = 0.002140
   Speedup = 0.766502
