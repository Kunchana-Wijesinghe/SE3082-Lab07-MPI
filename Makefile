.RECIPEPREFIX := >

CC = mpicc
CFLAGS = -Wall -Wextra -O2
MPIRUN = mpirun
MPI_FLAGS = --oversubscribe
NP = 4

PROGRAMS = \
	Exercise01/sum_bcast \
	Exercise02/sum_scatter \
	Exercise03/sum_gather \
	Exercise04/sum_reduce \
	Exercise05/sum_allreduce \
	Exercise06/sum_scan

.PHONY: all run clean

all: $(PROGRAMS)

Exercise01/sum_bcast: Exercise01/sum_bcast.c
>$(CC) $(CFLAGS) -o $@ $<

Exercise02/sum_scatter: Exercise02/sum_scatter.c
>$(CC) $(CFLAGS) -o $@ $<

Exercise03/sum_gather: Exercise03/sum_gather.c
>$(CC) $(CFLAGS) -o $@ $<

Exercise04/sum_reduce: Exercise04/sum_reduce.c
>$(CC) $(CFLAGS) -o $@ $<

Exercise05/sum_allreduce: Exercise05/sum_allreduce.c
>$(CC) $(CFLAGS) -o $@ $<

Exercise06/sum_scan: Exercise06/sum_scan.c
>$(CC) $(CFLAGS) -o $@ $<

run: all
>@for program in $(PROGRAMS); do \
		echo "========================================"; \
		echo "Running $$program with $(NP) processes"; \
		echo "========================================"; \
		$(MPIRUN) $(MPI_FLAGS) -np $(NP) ./$$program; \
		echo; \
	done

clean:
>rm -f $(PROGRAMS)
