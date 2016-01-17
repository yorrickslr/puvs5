// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //  Füge Headerdateien von OpenCL hinzu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          //	Definiere für Wort DATA_SIZE die Größe 10
#define MEM_SIZE    DATA_SIZE * sizeof(float)   //  Groesse der Daten im Speicher

/** Kernel Quelltext **/ 
const char *KernelSource =
	"#define DATA_SIZE 10												\n"
	"__kernel void test(__global float *input, __global float *output)  \n"
	"{																	\n"
	"	size_t i = get_global_id(0);									\n"
	"	output[i] = input[i] * input[i];								\n"
	"}																	\n"
	"\n";

/** **/
int main (void)
{
	cl_int				err;                      // Erschaffe Instanz eines besonderen integer/Typ, der von OpenCL definiert ist und zum Testen auf Fehler genutzt wird
	cl_platform_id*		platforms = NULL;         // Pointer auf OpenCL-Typ der PlattformID definiert, zeigt erst auf NULL
	char			    platform_name[1024];      // Erschaffe Char-Array der Namen der Plattform speichert
	cl_device_id	    device_id = NULL;         // Erschaffe Instanz des Typs der, die device_id speichert
	cl_uint			    num_of_platforms = 0,     // Erschaffe zwei Instanzen eines besonderen vorzeichenlosen integer/Typ, der von OpenCL definiert ist
					    num_of_devices = 0;       // einer speichert Anzahl an Plattformen anderer Anzahl an Geräten
	cl_context 			context;                  // Instanz für Kontext, wird genutzt um sich um Memory, command-queue und andere Objekte zu kümmern
	cl_kernel 			kernel;                   // Instanz (eines spez. OpenCL-Typs) für Kernel
	cl_command_queue	command_queue;            // Instanz (eines spez. OpenCL-Typs) für die Reihe der anzuwendenden Befehle
	cl_program 			program;                  // Instanz für das Programm 
	cl_mem				input, output;            // Speichert Input und Output
	float				data[DATA_SIZE] =         // float-Array mit 10 Elementen mit den angegeben Werten in der Reihenfolge
							{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	size_t				global[1] = {DATA_SIZE};  // Array mit einem Element smit Wert DATA_SIZE(10)
	float				results[DATA_SIZE] = {0}; //float-Array mit 10 Elementen mit dem Wert 0 

	/* 1) Abrufen und Auswahlen der zu verwendenden Geräte + Erstellen einer Befehlswarteschlange und Öffnen eines OpenCL-Kontexts und kopilieren des Kernels und Erstellung des Programms */

	// Teste ob Plattformen da sind und speicher Anzahl der und teste nach Aufrufen der dafür nötigen Funktion auf Fehler
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// // Teste ob Plattformen da sind und teste nach Aufrufen der dafür nötigen Funktion auf Fehler
	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// Gehe durch alle vorhandenen Plattformen durch
		for (unsigned int i=0; i<num_of_platforms; i++)
		{
			//Speicher Informationen über momentane Plattform i
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name,	NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			// Wenn im Namen der momentanen Plattform NVIDIA vorkommt
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// Speicher GeräteID und teste auf Fehler und gib notfalls Fehler aus
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// Speichert Kontext in oben definierten Variablen testet auf Fehler bei Funktionsaufruf
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// Erzeugen einer Befehlswarteschlange in oben definierten Variablen + testet auf Fehler bei Funktionsaufruf
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// Erzeugt Programm aus Kernelquelltext und speichert es in Programm Variable + Fehlercheck
	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

  //Kompiliere und Linke Programm aus den oben gespeicherten Daten und mache einen Fehlercheck
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	//
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) Zuweisen von Speicher zu Ein-und Ausgabe */

	// Speicher in Input einen OpenCL Buffer zum Lesen und in Output einen Buffer zum Schreiben
	input  = clCreateBuffer (context, CL_MEM_READ_ONLY,	 MEM_SIZE, NULL, &err);
	output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// Reiht Befehle in Schlange ein um in ein Bufferobjekt zu schreiben, kopiert von Hauptspeicher 'data' in buffer input
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	// Reihenfolge der Argumente des Kernels werden bestimmt
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3) Starten der Rechenkern Ausführung und Sammeln der Ergebnisse */

	//  Lass Den Berechnungskernel  in BefehlswarteSchlangeeinreihen 
	clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// blockiert den aufrufenden Thread bis alle OpenCL Aufrufe vollendet sind
	clFinish(command_queue);

	// Kopiere Ergebnisse des out buffers in results Array
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

  // Gib die in results gespeicherten Werte aus
  for (unsigned int i=0; i < DATA_SIZE; i++)
    printf("%f\n", results[i]);


	/* 4) Gib Speicher der in oben definierten Objekten frei */
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
/*
Ausgabe ist: [Visual Studio 2015 Community Edition unter Windows 8.1 aktuelle NVIDIA CUDA Version]
1.000000
4.000000
9.000000
16.000000
25.000000
36.000000
49.000000
64.000000
81.000000
100.000000
*/