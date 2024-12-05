CC = nvcc
TARGET = prog
OMP = -Xcompiler -fopenmp

all: $(TARGET)

$(TARGET): battle_sort.cu
	$(CC) $(OMP) -o $(TARGET) battle_sort.cu

clean:
	rm -f $(TARGET)
