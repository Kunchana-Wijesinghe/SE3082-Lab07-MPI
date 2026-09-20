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

    /*
     * Only root allocates the complete array.
     * Other processes keep full_array as NULL.
     */
    int *full_array = NULL;

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

    /*
     * Every process allocates memory only for its own chunk.
     */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    if (local_chunk == NULL)
    {
        printf("Rank %d: Local-chunk memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double start = MPI_Wtime();

    /*
     * Root divides full_array into equal chunks.
     * Each process receives one chunk.
     */
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
     * Exercise 2 keeps the manual Send/Recv collection.
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

            MPI_Recv(
                &received_sum,
                1,
                MPI_LONG_LONG,
                process,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            total_sum += received_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;

        printf("\n[Scatter] Total sum = %lld\n", total_sum);
        printf("[Scatter] Expected  = %lld\n", expected);
        printf("[Scatter] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Scatter] Time      = %.6f sec\n", elapsed);
    }

    free(local_chunk);

    if (rank == 0)
    {
        free(full_array);
    }

    MPI_Finalize();
    return 0;
}
