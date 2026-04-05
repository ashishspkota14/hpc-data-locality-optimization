/*
 * Data Locality Optimization: Cache-Friendly vs Cache-Unfriendly Data Structures
 * Based on: Azad et al. (2023) HPC Performance Bugs Study
 *
 * Compares:
 * 1. Linked list traversal (pointer chasing - cache unfriendly)
 * 2. Array sequential access (contiguous - cache friendly)
 * 3. Structure of Arrays (SoA) - maximum cache utilization
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define CUTOFF 5.0
#define CUTOFF_SQ (CUTOFF * CUTOFF)
#define TRIALS 5

#ifdef __MACH__
#include <mach/mach_time.h>
double get_time_ms()
{
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0)
        mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (double)t * info.numer / info.denom / 1e6;
}
#else
double get_time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ── Approach 1: Linked List (Cache-Unfriendly) ── */
typedef struct Node
{
    double x, y, z;
    double fx, fy, fz;
    int id;
    char padding[56]; /* simulate real object with extra fields */
    struct Node *next;
} Node;

/*
 * Create linked list with scattered memory allocation.
 * Allocate nodes one-by-one in shuffled order so they end up
 * at non-contiguous heap addresses (simulating real-world fragmentation).
 */
void create_linked_list(Node **nodes_out, int **order_out, int n, double box)
{
    int i, j, tmp;
    Node **ptrs = (Node **)malloc(n * sizeof(Node *));
    int *order = (int *)malloc(n * sizeof(int));

    if (!ptrs || !order)
    {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    /* Create shuffle order */
    for (i = 0; i < n; i++)
        order[i] = i;
    for (i = n - 1; i > 0; i--)
    {
        j = rand() % (i + 1);
        tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    /* Allocate each node individually in shuffled order */
    srand(42);
    for (i = 0; i < n; i++)
    {
        int idx = order[i];
        ptrs[idx] = (Node *)malloc(sizeof(Node));
        if (!ptrs[idx])
        {
            fprintf(stderr, "malloc node failed\n");
            exit(1);
        }
        ptrs[idx]->x = ((double)rand() / RAND_MAX) * box;
        ptrs[idx]->y = ((double)rand() / RAND_MAX) * box;
        ptrs[idx]->z = ((double)rand() / RAND_MAX) * box;
        ptrs[idx]->fx = 0;
        ptrs[idx]->fy = 0;
        ptrs[idx]->fz = 0;
        ptrs[idx]->id = idx;
        ptrs[idx]->next = NULL;
    }

    /* Link them sequentially (but memory is scattered) */
    for (i = 0; i < n - 1; i++)
    {
        ptrs[i]->next = ptrs[i + 1];
    }

    *nodes_out = ptrs[0];               /* return head */
    *order_out = (int *)((void *)ptrs); /* reuse ptrs array for cleanup */
    free(order);
}

void free_linked_list(Node *head)
{
    Node *cur = head;
    while (cur)
    {
        Node *tmp = cur;
        cur = cur->next;
        free(tmp);
    }
}

long compute_forces_linked(Node *head)
{
    long interactions = 0;
    Node *pi, *pj;

    /* Zero forces */
    for (pi = head; pi != NULL; pi = pi->next)
    {
        pi->fx = 0;
        pi->fy = 0;
        pi->fz = 0;
    }

    /* Pairwise force - pointer chasing through scattered memory */
    for (pi = head; pi != NULL; pi = pi->next)
    {
        for (pj = pi->next; pj != NULL; pj = pj->next)
        {
            double dx = pi->x - pj->x;
            double dy = pi->y - pj->y;
            double dz = pi->z - pj->z;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < CUTOFF_SQ && dist_sq > 0)
            {
                double inv = 1.0 / sqrt(dist_sq);
                double f = inv * inv * inv;
                pi->fx += f * dx;
                pi->fy += f * dy;
                pi->fz += f * dz;
                pj->fx -= f * dx;
                pj->fy -= f * dy;
                pj->fz -= f * dz;
                interactions++;
            }
        }
    }
    return interactions;
}

/* ── Approach 2: Array of Structures (Contiguous) ── */
typedef struct
{
    double x, y, z;
    double fx, fy, fz;
    int id;
    char padding[56];
} ParticleAoS;

long compute_forces_aos(ParticleAoS *p, int n)
{
    long interactions = 0;
    int i, j;
    for (i = 0; i < n; i++)
    {
        p[i].fx = 0;
        p[i].fy = 0;
        p[i].fz = 0;
    }
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < CUTOFF_SQ && dist_sq > 0)
            {
                double inv = 1.0 / sqrt(dist_sq);
                double f = inv * inv * inv;
                p[i].fx += f * dx;
                p[i].fy += f * dy;
                p[i].fz += f * dz;
                p[j].fx -= f * dx;
                p[j].fy -= f * dy;
                p[j].fz -= f * dz;
                interactions++;
            }
        }
    }
    return interactions;
}

/* ── Approach 3: Structure of Arrays (Maximum Cache Efficiency) ── */
long compute_forces_soa(double *x, double *y, double *z,
                        double *fx, double *fy, double *fz, int n)
{
    long interactions = 0;
    int i, j;
    memset(fx, 0, n * sizeof(double));
    memset(fy, 0, n * sizeof(double));
    memset(fz, 0, n * sizeof(double));
    for (i = 0; i < n; i++)
    {
        double xi = x[i], yi = y[i], zi = z[i];
        double fxi = 0, fyi = 0, fzi = 0;
        for (j = i + 1; j < n; j++)
        {
            double dx = xi - x[j]; /* sequential array access */
            double dy = yi - y[j];
            double dz = zi - z[j];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < CUTOFF_SQ && dist_sq > 0)
            {
                double inv = 1.0 / sqrt(dist_sq);
                double f = inv * inv * inv;
                fxi += f * dx;
                fyi += f * dy;
                fzi += f * dz;
                fx[j] -= f * dx;
                fy[j] -= f * dy;
                fz[j] -= f * dz;
                interactions++;
            }
        }
        fx[i] += fxi;
        fy[i] += fyi;
        fz[i] += fzi;
    }
    return interactions;
}

int main(void)
{
    int sizes[] = {500, 1000, 2000, 3000, 5000, 7000, 10000};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    double box = 30.0;
    int si, t, i;

    printf("====================================================================\n");
    printf("DATA LOCALITY BENCHMARK (C): Linked List vs Array vs SoA\n");
    printf("====================================================================\n");
    printf("%-8s %12s %12s %12s %10s %10s\n",
           "N", "Linked(ms)", "AoS(ms)", "SoA(ms)", "AoS Spdup", "SoA Spdup");
    printf("--------------------------------------------------------------------\n");

    for (si = 0; si < nsizes; si++)
    {
        int n = sizes[si];

        /* ── Linked list benchmark ── */
        double best_ll = 1e9;
        for (t = 0; t < TRIALS; t++)
        {
            Node *head = NULL;
            int *cleanup = NULL;
            create_linked_list(&head, &cleanup, n, box);

            double t0 = get_time_ms();
            compute_forces_linked(head);
            double dt = get_time_ms() - t0;
            if (dt < best_ll)
                best_ll = dt;

            free_linked_list(head);
            free(cleanup); /* free the ptrs array */
        }

        /* ── AoS benchmark ── */
        ParticleAoS *aos = (ParticleAoS *)malloc(n * sizeof(ParticleAoS));
        if (!aos)
        {
            fprintf(stderr, "malloc aos failed\n");
            return 1;
        }
        srand(42);
        for (i = 0; i < n; i++)
        {
            aos[i].x = ((double)rand() / RAND_MAX) * box;
            aos[i].y = ((double)rand() / RAND_MAX) * box;
            aos[i].z = ((double)rand() / RAND_MAX) * box;
            aos[i].id = i;
        }
        double best_aos = 1e9;
        for (t = 0; t < TRIALS; t++)
        {
            double t0 = get_time_ms();
            compute_forces_aos(aos, n);
            double dt = get_time_ms() - t0;
            if (dt < best_aos)
                best_aos = dt;
        }
        free(aos);

        /* ── SoA benchmark ── */
        double *sx = (double *)malloc(n * sizeof(double));
        double *sy = (double *)malloc(n * sizeof(double));
        double *sz = (double *)malloc(n * sizeof(double));
        double *sfx = (double *)malloc(n * sizeof(double));
        double *sfy = (double *)malloc(n * sizeof(double));
        double *sfz = (double *)malloc(n * sizeof(double));
        if (!sx || !sy || !sz || !sfx || !sfy || !sfz)
        {
            fprintf(stderr, "malloc soa failed\n");
            return 1;
        }
        srand(42);
        for (i = 0; i < n; i++)
        {
            sx[i] = ((double)rand() / RAND_MAX) * box;
            sy[i] = ((double)rand() / RAND_MAX) * box;
            sz[i] = ((double)rand() / RAND_MAX) * box;
        }
        double best_soa = 1e9;
        for (t = 0; t < TRIALS; t++)
        {
            double t0 = get_time_ms();
            compute_forces_soa(sx, sy, sz, sfx, sfy, sfz, n);
            double dt = get_time_ms() - t0;
            if (dt < best_soa)
                best_soa = dt;
        }
        free(sx);
        free(sy);
        free(sz);
        free(sfx);
        free(sfy);
        free(sfz);

        /* Print results */
        double sp_aos = best_ll / best_aos;
        double sp_soa = best_ll / best_soa;
        printf("%-8d %12.2f %12.2f %12.2f %9.2fx %9.2fx\n",
               n, best_ll, best_aos, best_soa, sp_aos, sp_soa);
    }

    printf("====================================================================\n");
    return 0;
}