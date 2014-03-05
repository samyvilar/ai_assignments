//
//  main.c
//  n_queens
//
//  Created by Samy Vilar on 3/4/14.
//  Copyright (c) 2014 samyvilar. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Calculate the collisions over the left and right diaganols of a 2-D chess board filled with 0s and 1s
unsigned int collisions(unsigned char *board, unsigned int board_size)
{
    unsigned int diaganol_mag = board_size;
    // keep track of diagnol sums by mag each board has board_size sizes ...
    unsigned int (*diaganol_sums)[4] = calloc(board_size, sizeof(unsigned int[4]));
    unsigned int *current_diaganols = diaganol_sums[board_size - 1];

    unsigned char *current_row = board + (board_size * board_size); // last row ...
    while (diaganol_mag--) // center diagnols ...
    {
        current_row -= board_size;
        current_diaganols[1] += current_row[(board_size - diaganol_mag + 1)];
        current_diaganols[0] += current_row[diaganol_mag];
    }

    unsigned int diaganols = board_size;
    while (diaganols--)
    {
        diaganol_mag = diaganols;
        current_diaganols = diaganol_sums[diaganol_mag - 1];
        current_row = board + (board_size * board_size); // last row ...
        while (diaganol_mag--) // off-center diagnols ...
        {
            current_row          -= board_size;
            current_diaganols[0] += current_row[diaganol_mag];
            current_diaganols[1] += current_row[board_size - diaganol_mag - 1];
            current_diaganols[2] += board[(diaganol_mag * board_size) + (diaganols - diaganol_mag - 1)];
            current_diaganols[3] += board[(diaganol_mag * board_size) + board_size - (diaganols - diaganol_mag)];

        }
    }

    #define sum_collisions diaganol_mag
    sum_collisions = 0;     // skip diagnols of size 1, since they can't have collisions ...
    while (--board_size)  // go over all diagnols, for each approproate size calculate collision to their sum - 1
    {
        current_diaganols = diaganol_sums[board_size];
        sum_collisions += (current_diaganols[0] - (current_diaganols[0] != 0)) // subtract 1 if non-zero ...
                        + (current_diaganols[1] - (current_diaganols[1] != 0))
                        + (current_diaganols[2] - (current_diaganols[2] != 0))
                        + (current_diaganols[3] - (current_diaganols[3] != 0));
    }

    free(diaganol_sums);
    return sum_collisions;
    #undef sum_collisions
}
