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

// Calculate the collisions over the left and right diagonals of a 2-D chess board filled with 0s and 1s
unsigned int collisions(unsigned char *board, unsigned int board_size)
{
    unsigned int
        diagonal_mag = board_size,
        (*diagonal_sums)[4] = calloc(board_size, sizeof(unsigned int[4])),  // keep track of diagnol sums by mag each board has board_size sizes ...
        *current_diagonals = diagonal_sums[board_size - 1];

    unsigned char *top_off_center_rows = (board + (board_size * board_size)), *bottom_off_center_rows;
    while (diagonal_mag--) // do center diagnols first since theres only two of them ...
    {
        top_off_center_rows -= board_size;
        current_diagonals[1] += top_off_center_rows[board_size - diagonal_mag - 1]; // left ...
        current_diagonals[0] += top_off_center_rows[diagonal_mag]; // right
    }

    unsigned int diagonals = board_size;
    while (diagonals--)  // go over all the remaining off center diagonals, theres 4 for each magnitude ...
    {
        diagonal_mag = diagonals;
        current_diagonals = diagonal_sums[diagonal_mag - 1];

        bottom_off_center_rows   = board + (board_size * board_size);   // bottom rows
        top_off_center_rows      = board + (diagonal_mag * board_size); // top rows

        while (diagonal_mag--) // do off-center diagnols, off-center above and below in each direction ...
        {
            bottom_off_center_rows  -= board_size; // bottom off-center
            current_diagonals[0]    += bottom_off_center_rows[diagonal_mag]; // right
            current_diagonals[1]    += bottom_off_center_rows[board_size - diagonal_mag - 1]; // left

            top_off_center_rows     -= board_size; // top off-center ...
            current_diagonals[2]    += top_off_center_rows[(diagonals - diagonal_mag - 1)]; // left
            current_diagonals[3]    += top_off_center_rows[(board_size - (diagonals - diagonal_mag))]; // right
        }
    }

    #define sum_collisions diagonal_mag
    sum_collisions = 0;
    while (--board_size)  // skip diagonals of size 1, since they can't have collisions ...
    {   // go over all diagonals, for each approproate size calculate collision to their sum - 1
        current_diagonals = diagonal_sums[board_size];
        sum_collisions  += (current_diagonals[0] - (current_diagonals[0] != 0)) // subtract 1 if non-zero ...
                        +  (current_diagonals[1] - (current_diagonals[1] != 0))
                        +  (current_diagonals[2] - (current_diagonals[2] != 0))
                        +  (current_diagonals[3] - (current_diagonals[3] != 0));
    }

    free(diagonal_sums);
    return sum_collisions;
    #undef sum_collisions
}
