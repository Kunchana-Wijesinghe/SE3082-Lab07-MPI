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

    /* Every process allocates only its own chunk. */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    if (local_chunk == NULL)
    {
        printf("Rank %d: Local-chunk memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
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

    /*
     * MPI_Scan calculates a prefix sum.
     * Each rank receives the sum from rank 0 up to its own rank.
     */
    long long prefix_sum = 0;

    MPI_Scan(
        &local_sum,
        &prefix_sum,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    long long sum_before_me = prefix_sum - local_sum;

    /*
     * For values 1 to K, the expected prefix sum is K * (K + 1) / 2.
     */
    long long k = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = k * (k + 1) / 2;

    printf(
        "Rank %d: local_sum = %lld, prefix_sum = %lld, "
        "sum_before_me = %lld, prefix_correct = %s\n",
        rank,
        local_sum,
        prefix_sum,
        sum_before_me,
        prefix_sum == expected_prefix ? "YES" : "NO"
    );

    /*
     * The last rank's prefix sum must equal the global total.
     */
    if (rank == size - 1)
    {
        double elapsed = MPI_Wtime() - start;
        long long expected_total = (long long)N * (N + 1) / 2;

        printf("\n[Scan] Last rank prefix sum = %lld\n", prefix_sum);
        printf("[Scan] Expected total       = %lld\n", expected_total);
        printf("[Scan] Correct?             = %s\n",
               prefix_sum == expected_total ? "YES" : "NO");
        printf("[Scan] Time                 = %.6f sec\n", elapsed);
    }

    free(local_chunk);

    if (rank == 0)
    {
        free(full_array);
    }

    MPI_Finalize();
    return 0;
}
