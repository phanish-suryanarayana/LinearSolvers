all: mpi_real

CC = mpicc

# CFLAGS = -O3 -funroll-loops -g -Wall -DDEBUG

# if using gcc
CFLAGS = -O3 -funroll-loops -g -Wall -lm

# if using intel, please use this CFLAGS
# CFLAGS = -O3 -funroll-loops -g -Wall

# SOURCEC = main.c system.c AAR.c tools.c
OBJSC = main.o system.o AAR.o tools.o

mpi_real: $(OBJSC)
	${CC} ${CFLAGS} ${LIBS} -o $@ $^ 

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

.PHONY: clean
clean:
	rm -f ./*.o *~ core core*
