#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0)
    {
        if (rank == 0)
        {
            printf("Error: Number of processes must evenly divide %d.\n", N);
        }

        MPI_Finalize();
        return 1;
    }

    /*
     * Every process allocates the complete array.
     * This is the memory disadvantage of MPI_Bcast.
     */
    int *array = (int *)malloc(N * sizeof(int));

    if (array == NULL)
    {
        printf("Rank %d: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Root process fills the array with values from 1 to N. */
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    /*
     * Root broadcasts the complete array to every process.
     * Every process must participate in this collective operation.
     */
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;

    long long local_sum = 0;

    for (int i = start_idx; i < end_idx; i++)
    {
        local_sum += array[i];
    }

    printf("Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    /*
     * Non-root processes send their partial sums to root.
     * Root receives each value and calculates the total.
     */
    if (rank != 0)
    {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        long long total_sum = local_sum;

        for (int process = 1; process < size; process++)
        {
            long long received_sum;

            MPI_Recv(&received_sum,
                     1,
                     MPI_LONG_LONG,
                     process,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            total_sum += received_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;

        printf("\n[Bcast] Total sum = %lld\n", total_sum);
        printf("[Bcast] Expected  = %lld\n", expected);
        printf("[Bcast] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time      = %.6f sec\n", elapsed);
    }

    free(array);

    MPI_Finalize();
    return 0;
}
