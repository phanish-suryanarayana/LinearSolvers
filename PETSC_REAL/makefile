all:system

CPPFLAGS = -I ./src -g -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
# CPPFLAGS = -I ./src -g -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DDEBUG
 
SOURCEC = ./main.c system.c tools.c AAR_Real.c 
OBJSC = ./main.o system.o tools.o AAR_Real.o 
LIBBASE = ./petsc_real

# For PETSC 3.11.3
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# For PETSC 3.5.3
# include ${PETSC_DIR}/conf/variables
# include ${PETSC_DIR}/conf/rules

system: ${OBJSC} chkopts
	${CLINKER} -o ${LIBBASE} ${OBJSC} ${PETSC_LIB} 
	${RM} ${OBJSC}



