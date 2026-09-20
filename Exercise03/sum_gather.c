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

    int chunk_size = N / size;
    int *full_array = NULL;

    /* Only root allocates and fills the complete array. */
    if (rank == 0)
    {
        full_array = (int *)malloc(N * sizeof(int));

        if (full_array == NULL)
        {
            printf("Root: Full-array memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++)
        {
            full_array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Every process allocates memory only for its own chunk. */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    if (local_chunk == NULL)
    {
        printf("Rank %d: Local-chunk memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
     * Only root needs an array to receive all partial sums.
     */
    long long *all_sums = NULL;

    if (rank == 0)
    {
        all_sums = (long long *)malloc(size * sizeof(long long));

        if (all_sums == NULL)
        {
            printf("Root: all_sums memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double start = MPI_Wtime();

    MPI_Scatter(
        full_array,
        chunk_size,
        MPI_INT,
        local_chunk,
        chunk_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    long long local_sum = 0;

    for (int i = 0; i < chunk_size; i++)
    {
        local_sum += local_chunk[i];
    }

    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;

    printf("Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    /*
     * Gather one local_sum from every process.
     * On root, the values are stored in rank order in all_sums.
     */
    MPI_Gather(
        &local_sum,
        1,
        MPI_LONG_LONG,
        all_sums,
        1,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        long long total_sum = 0;

        for (int process = 0; process < size; process++)
        {
            total_sum += all_sums[process];
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;

        printf("\n[Gather] Partial sums in rank order:\n");

        for (int process = 0; process < size; process++)
        {
            printf("Rank %d partial sum = %lld\n",
                   process, all_sums[process]);
        }

        printf("\n[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Expected  = %lld\n", expected);
        printf("[Gather] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Gather] Time      = %.6f sec\n", elapsed);
    }

    free(local_chunk);

    if (rank == 0)
    {
        free(full_array);
        free(all_sums);
    }

    MPI_Finalize();
    return 0;
}
