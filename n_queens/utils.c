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
#include <time.h>

#include <limits.h>

#if   (UINTPTR_MAX == UINT64_MAX)
#define word_t long long int
#define word_t_size 8
#elif (UINTPTR_MAX == UINT32_MAX)
    #define word_t int
    #define word_t_size 4
#elif (UINTPTR_MAX == UINT16_MAX)
    #define word_t short
    #define word_t_size 2
#elif (UINTPTR_MAX == UINT8_MAX)
    #define word_t char
    #define word_t_size 1
#else
    #error "Unable to determing intergral type for pointer type"
#endif


#define swap_xor(a, b) ((a ^= b), (b ^= a), (a ^= b))

#define instr_postfix_1 "b" // byte
#define instr_postfix_2 "w" // short
#define instr_postfix_4 "l" // int
#define instr_postfix_8 "q" // long
#define get_instr_postfix(size) instr_postfix_ ## size
#define att_instr(instr_name, operand_size) instr_name get_instr_postfix(operand_size)


#define swap_xchg_1(a, b)  ({asm(att_instr("xchg", 1) " %0, %1" : "=r"(a), "=r"(b) : "0"(a), "1"(b));})
#define swap_xchg_2(a, b)  ({asm(att_instr("xchg", 2) " %0, %1" : "=r"(a), "=r"(b) : "0"(a), "1"(b));})
#define swap_xchg_4(a, b)  ({asm(att_instr("xchg", 4) " %0, %1" : "=r"(a), "=r"(b) : "0"(a), "1"(b));})
#define swap_xchg_8(a, b)  ({asm(att_instr("xchg", 8) " %0, %1" : "=r"(a), "=r"(b) : "0"(a), "1"(b));})
//#define swap_4_xchng xor_swap

#define instr(instr, size) instr ## size

#define swap_xchg(a, b) (\
    {switch(sizeof(a)) {\
        case 1: instr(swap_xchg_, 1)(a, b); break ;\
        case 2: instr(swap_xchg_, 2)(a, b); break ;\
        case 4: instr(swap_xchg_, 4)(a, b); break ;\
        case 8: instr(swap_xchg_, 8)(a, b); break ;}})


#define swap swap_xchg


#define bit_size(a) (sizeof(a) * CHAR_BIT)

#define uint1byt unsigned char
#define uint2byt unsigned short
#define uint4byt unsigned int
#define uint8byt unsigned long long int


#define sint1byt signed char
#define sint2byt signed short
#define sint4byt signed int
#define sint8byt signed long long int

#define uint_byt_t(_byt_mag) uint ## _byt_mag ## byt
#define sint_byt_t(_byt_mag) sint ## _byt_mag ## byt

#define instr_postfix_1 "b" // byte
#define instr_postfix_2 "w" // short
#define instr_postfix_4 "l" // int
#define instr_postfix_8 "q" // long
#define get_instr_postfix(size) instr_postfix_ ## size
#define att_instr(instr_name, operand_size) instr_name get_instr_postfix(operand_size)

#define intrsc_signat(ret_type) __inline ret_type __attribute__((__gnu_inline__, __always_inline__, __artificial__))
#ifdef __INTEL_COMPILER // icc doesn't have __builtin_clzs or __builtin_ctz which is kind of strange
    intrsc_signat(unsigned) __builtin_clzs(unsigned short x) { // count leading zeros, return 15 if all zeros ...
        asm     (att_instr("bsr", 2) " %0, %0\t\nxor $15, %0\t\n" : "=r" (x) : "0"(x));
        return x;
    }

    intrsc_signat(unsigned) __builtin_ctzs(unsigned short x) { // count trailing zeros, return 0 if 0 ...
        asm (att_instr("bsf", 2) " %0, %0" : "=r" (x) : "0"(x));
        return x;
    }

    intrsc_signat(unsigned) __builtin_popcounts(unsigned short x) { // count set 1s return 0 if 0
        return __builtin_popcount(x) - 8;
    }
#endif

// there is no instruction that supports bit scan on a char ...

#define __builtin_clzb(x) (cnt_leadn_zrs_16((unsigned char)(x)) ^ 8) // subtract leading 8 zeros ...
#define __builtin_ctzb(x) cnt_trlng_zrs_16((unsigned char)(x))
#define __builtin_popcountb(x) bit_ones_cnt_16((unsigned char)(x))


#define cnt_leadn_zrs_8 __builtin_clzb
#define cnt_leadn_zrs_16 __builtin_clzs
#define cnt_leadn_zrs_32 __builtin_clz
#define cnt_leadn_zrs_64 __builtin_clzll


#define cnt_leadn_zrs(x) ({\
    int zero_cnt; \
    switch(sizeof(x)){\
        case 1: zero_cnt = cnt_leadn_zrs_8(x); break; \
        case 2: zero_cnt = cnt_leadn_zrs_16(x); break ; \
        case 4: zero_cnt = cnt_leadn_zrs_32(x); break ; \
        case 8: zero_cnt = cnt_leadn_zrs_64(x); break ; \
   } zero_cnt; })

#define leadn_one_index(x) (bit_size(x) - cnt_leadn_zrs(x))

#define _rshift_(a, _mag, _type) ({                                             \
    typeof(a) _r = (a);                                                         \
    switch (sizeof(a)) {                                                        \
        case 1: _r = (_type(1))_r >> (_mag); break ;                            \
        case 2: _r = (_type(2))_r >> (_mag); break ;                            \
        case 4: _r = (_type(4))_r >> (_mag); break ;                            \
        case 8: _r = (_type(8))_r >> (_mag); break ;                            \
    } _r; })


#define rshift_logcl(a, _mag) _rshift_(a, _mag, uint_byt_t) // arithmetic right shift (extending sign)
#define rshift_airth(a, _mag) _rshift_(a, _mag, sint_byt_t) // logical right shift (shifts in zeros)

#define sign_bit_ext(a) rshift_airth(a, (bit_size(a) - 1))  // extends the sign bit creating a mask of ones or zeros..

#define sign_bit_bool(a) rshift_logcl(a, (bit_size(a) - 1)) // returns 0 if sign bit is 0 or 1 if sign bits is 1 (so it just moves the sign from ms loc to ls loc ...)

#define bool_by_bit_shft(a) ({typeof(a) _sb = (a); rshift_logcl(_sb, leadn_one_index(_sb));})
#define bool_by_logical(a) (!!(a))

// returns the absolute value of (a) assumes (a) is signed and underlying bit repr is twos complement
// -a == ~-a + 1 == (-a ^ -1) + 1
#define abs_by_sign_bit(a) ({typeof(a) _val = (a); ((_val ^ sign_bit_ext(_val)) + sign_bit_bool(_val)); })
#define abs_by_cmp(a) (((a) < 0) ? (((a) ^ (typeof(a))-1) + (typeof(a))1) : (a))

#define abs abs_by_sign_bit

// two bit sequences equal if the a == b => bool(a ^ b) ^ 1

#define not_eq_by_xor(a, b) bool_by_bit_shft((a) ^ (b))
#define eq_by_xor(a, b) (bool_by_bit_shft((a) ^ (b)) ^ (typeof(a))1)

#define eq_by_cmp(a, b) ((a) == (b))

#define eq_bool eq_by_xor

#define half(a)     rshift_logcl(a, 1)
#define quarter(a)  rshift_logcl(a, 2)


#define min_by_cmp(a, b)        (((a) < (b)) ? (a) : (b))
#define max_by_cmp(a, b)        (((a) < (b)) ? (b) : (a))
// branchless min/max counterparts ...

#define min_by_subt(a, b)    	((b) + (((a) - (b)) & sign_bit_ext((a) - (b))))
#define max_by_subt(a, b)		((a) + (((b) - (a)) & sign_bit_ext((a) - (b))))

#define min_by_xor(a, b)    	((b) ^ (((a) ^ (b)) & sign_bit_ext((a) - (b))))
#define max_by_xor(a, b)        ((a) ^ (((a) ^ (b)) & sign_bit_ext((a) - (b))))

#define min min_by_xor
#define max max_by_xor


#define unlikely(expr) __builtin_expect(expr, 0)
#define likely(expr) __builtin_expect(expr, 1)


#define timed(func) ({                              \
    clock_t __start__, __end__;                     \
    __start__ = clock();                            \
        func();                                     \
    __end__ = clock();                              \
    ((double)(__end__ - __start__))/CLOCKS_PER_SEC; \
})


#define perm_elem_type unsigned char

// diaganol collision occur between two points when there |delta(x cords)| == |delta(y cords)|
// ie the diff in their corresponding x and y coordinates are the same
// and theres no other queen in between them ....
// there can never be more than n - 1 diagonal collision in any giving board of size n with n queens ...
// assuming there are no row or column collisions (col_cords has unique values)
// there are a total of 4n diagnols in each board of size n
// a queen may either be in (y cord - x cord) diag and/or the ((n - y cord) - x cord) diag respectively
unsigned int diag_collisions(char *col_cords, unsigned int board_size) {
    long long
        index,
        cnts[4 * board_size],
        sum = 0;

    memset(cnts, 0, sizeof(cnts));

    typeof(cnts[0])
        *cols_r = &cnts[3 * board_size],
        *cols_l = &cnts[board_size];

    for (index = 0; index < board_size; index++) { // count the number of queens in each diagnol
        cols_r[col_cords[index] - index]++;
        cols_l[(board_size - col_cords[index]) - index]++;
    }

    for (index = 0; index < sizeof(cnts)/sizeof(cnts[0]); index++)
        if (cnts[index])
            sum += cnts[index] - 1;
//        sum += max(cnts[index], 1) - 1;

    return sum;

//    unsigned word_t
//        row_index, col_index, diag_col_cnt = 0;
//
//    for (row_index = 0; row_index < board_size; row_index++) // x coordinates ...
//        for (col_index = 0; col_index < board_size; col_index++) // y coordinates ...
//            diag_col_cnt += eq_bool(
//                abs(col_cords[row_index] - col_cords[col_index]),
//                abs(row_index - col_index)
//            );
//
//    return diag_col_cnt - board_size;

//    unsigned char
//        row_cnt = board_size,
//        col_cnt,
//        diag_col = 0,
//        index;
//
//    while (row_cnt--)
//    {
//        col_cnt = board_size;
//        while (col_cnt--)
//            diag_col
//              += ((col_cnt > row_cnt) ? (col_cnt - row_cnt) : (row_cnt - col_cnt))
//                    ==
//              (
//                (col_cords[row_cnt] > col_cords[col_cnt])
//              ? (col_cords[row_cnt] - col_cords[col_cnt])
//              : (col_cords[col_cnt] - col_cords[row_cnt])
//              );
//    }
//    return diag_col - board_size; // subtract all the points that collide with themselfs ...
}


// Calculate the collisions over the left and right diagonals of a 2-D chess board filled with 0s and 1s
// board contains the y coordinates ...
unsigned int collisions(unsigned char *board, unsigned int board_size) {
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


#define is_non_null(v) (v)
#define is_null(v) (!is_non_null(v))

#define word_type unsigned int
// Scapegoat binary tree, each node has their left child containing greater or equal values,
// right node containg smaller values ...
#define TREE_BLOCK_MAG 128
#define ALPHA    0.22 //0.288
#define EPSILON  0.005
typedef struct tree_type {
    struct tree_type
        *_left,
        *_right;

    unsigned _key, _weight;
} tree_type;

#define left_tree(t) (*(tree_type **)t)
#define set_left_tree(t, value) (left_tree(t) = (value))
#define right_tree(tree) (tree->_right)
#define set_right_tree(tree, value) (right_tree(tree) = (value))
#define is_empty_tree(tree) is_null(left_tree(tree))
#define is_non_leaf(tree) is_non_null(right_tree(tree))
#define key(tree) (tree->_key)
#define set_key(tree, value) (key(tree) = (value))
#define weight(tree) (tree->_weight)
#define set_weight(tree, value) (weight(tree) = (value))

tree_type *recycled_trees = NULL, *allocated_trees = (tree_type[TREE_BLOCK_MAG]){{}};
word_type available_trees = TREE_BLOCK_MAG;
tree_type *tree_stack[100] = {NULL}, **tree_stack_p = tree_stack;

#define push_tree(tree) (*tree_stack_p++ = (tree))
#define pop_tree() (*--tree_stack_p)
#define tree_stack_is_non_empty() (tree_stack_p != tree_stack)
#define tree_stack_cnt() (tree_stack_p - (tree_type **)tree_stack)

#define recycle_tree_macro(tree) ({set_left_tree(tree, recycled_trees); recycled_trees = tree;})
void recycle_tree(tree_type *tree) {
    recycle_tree_macro(tree);
}


#define allocate_tree_macro() ({                                                        \
    tree_type *_alloc_tree_;                                                            \
    if (is_non_null(recycled_trees))                                                    \
        (_alloc_tree_ = recycled_trees), (recycled_trees = left_tree(_alloc_tree_));    \
    else if (available_trees)                                                           \
        _alloc_tree_ = &allocated_trees[--available_trees];                             \
    else {                                                                              \
        allocated_trees = malloc(sizeof(allocated_trees[0]) * TREE_BLOCK_MAG);          \
        available_trees = TREE_BLOCK_MAG - 1;                                           \
        _alloc_tree_ = &allocated_trees[TREE_BLOCK_MAG - 1];                            \
    } _alloc_tree_; })

tree_type *allocate_tree() {
    return allocate_tree_macro();
}

#define new_tree(key, weight) ({                        \
    tree_type *_nw_tree_ = allocate_tree_macro();       \
    set_key(_nw_tree_, key);                            \
    set_weight(_nw_tree_, weight);                      \
    set_right_tree(_nw_tree_, NULL);                    \
    _nw_tree_; })

#define update_weight(tree) set_weight(tree, ((weight(left_tree(tree)) + weight(right_tree(tree)))))

//tree_type *__temp__;
//#define left_rotation_macro(tree)\
//    (                                                                                     \
//        (__temp__ = left_tree(tree)),/* save left tree */                                 \
//        xor_swap(key(tree), key(right_tree(tree))), /* swap root and left key  */       \
//        set_left_tree(tree, right_tree(tree)), /* save right tree */                      \
//        set_right_tree(tree, right_tree(right_tree(tree))), /* move sub right tree up */              \
//        set_right_tree(left_tree(tree), left_tree(left_tree(tree))),/* move right-left tree to left-right position */  \
//        set_left_tree(left_tree(tree), __temp__),  /* move left tree down*/     \
//        update_weight(left_tree(tree))                                          \
//   )
//#define left_rotation_macro(tree) ({                                  \
//    tree_type *l_tree = left_tree(tree), *r_tree = right_tree(tree);  \
//    typeof(key(tree)) root_key = key(tree);                           \
//                                                                      \
//    set_key(tree, key(r_tree));                                       \
//    set_key(r_tree, _key);                                            \
//                                                                      \
//    set_left_tree(tree, r_tree);                                      \
//    set_right_tree(tree, right_tree(r_tree));                         \
//    set_right_tree(r_tree, left_tree(r_tree));                        \
//                                                                      \
//    set_left_tree(r_tree, l_tree);                                    \
//    update_weight(r_tree);                                            \
//})

#define left_rotation_macro(tree) ({                                \
    typeof(tree) r_tree = right_tree(tree);                         \
    swap(key(tree),                 key(r_tree));                   \
    swap(left_tree(tree),           right_tree(tree));              \
    swap(right_tree(r_tree),        left_tree(r_tree));             \
    swap(right_tree(tree),          left_tree(left_tree(tree)));    \
    update_weight(r_tree);                                          \
})


//void left_rotation(tree_type *tree)
//{
//    tree_type *orig_left_tree = left_tree(tree);/* save left tree */
//    xor_swap(key(tree), key(right_tree(tree))); /* swap root and left key  */
//    set_left_tree(tree, right_tree(tree)); /* save right tree */
//
//    set_right_tree(tree, right_tree(right_tree(tree))); /* move sub right tree up */
//    set_right_tree(left_tree(tree), left_tree(left_tree(tree)));/* move right-left tree to left-right position */
//    set_left_tree(left_tree(tree), orig_left_tree);  /* move left tree down*/
//
//    update_weight(left_tree(tree));
//}
//


//#define right_rotation_macro(tree)                                                \
//    (                                                                       \
//        (__temp__ = right_tree(tree)),/* save right right tree*/                                      \
//        xor_swap(key(tree), key(left_tree(tree))), /* swap root and right keys ...*/ \
//        set_right_tree(tree, left_tree(tree)), /* save left tree*/                              \
//        set_left_tree(tree, left_tree(left_tree(tree))), /* move left sub tree up*/                    \
//        set_left_tree(right_tree(tree), right_tree(right_tree(tree))), /* move left-right tree to right-left position */      \
//        set_right_tree(right_tree(tree), __temp__), /* move right tree down */                         \
//        update_weight(right_tree(tree))                                    \
//    )
//
//void right_rotation(tree_type *tree)
//{
//    tree_type *orig_right_tree = right_tree(tree);/* save right right tree */
//    xor_swap(key(tree), key(left_tree(tree))); /* swap root and right keys ...*/
//    set_right_tree(tree, left_tree(tree)); /* save left tree */
//
//    set_left_tree(tree, left_tree(left_tree(tree))); /* move left sub tree up*/
//    set_left_tree(right_tree(tree), right_tree(right_tree(tree))); /* move left-right tree to right-left position */      \
//    set_right_tree(right_tree(tree), orig_right_tree); /* move right tree down */
//
//    update_weight(right_tree(tree));
//}

#define right_rotation_macro(tree) ({                           \
    typeof(tree) l_tree = left_tree(tree);                      \
    swap(key(tree),             key(l_tree));                   \
    swap(left_tree(tree),       right_tree(tree));              \
    swap(left_tree(l_tree),     right_tree(l_tree));            \
    swap(left_tree(tree),       right_tree(right_tree(tree)));  \
    update_weight(l_tree);                                      \
})

//#define right_rotation_macro(tree) ({                                   \
//    typeof(tree) l_tree = left_tree(tree), r_tree = right_tree(tree);   \
//    typeof(key(tree)) root_key = key(tree);                             \
//                                                                        \
//    set_key(tree, key(l_tree));                                         \
//    set_key(l_tree, root_key);                                          \
//                                                                        \
//    set_right_tree(tree, l_tree);                                       \
//    set_left_tree(tree, left_tree(l_tree));                             \
//    set_left_tree(l_tree, right_tree(l_tree));                          \
//                                                                        \
//    set_right_tree(l_tree, r_tree);                                     \
//    update_weight(l_tree);                                              \
//})

#define destroy_tree(tree) ({                                   \
    push_tree(tree);                                            \
    while (unlikely(tree_stack_is_non_empty())) {               \
        tree = pop_tree();                                      \
        if (is_non_leaf(tree)) {                                \
            push_tree(right_tree(tree));                        \
            push_tree(left_tree(tree));                         \
        }                                                       \
        recycle_tree_macro(tree);                               \
    }                                                           \
})

#define balance_weight_tree(tree) ({                                                         \
    double weighted_alpha;                                                                   \
    while (tree_stack_is_non_empty()) {                                                      \
        tree = pop_tree();                                                                   \
        update_weight(tree);                                                                 \
        weighted_alpha = weight(tree) * ALPHA;                                               \
        if (weight(right_tree(tree)) < weighted_alpha) {                                     \
            if (weight(left_tree(left_tree(tree))) <= ((ALPHA + EPSILON) * weight(tree))) {  \
                left_rotation_macro(left_tree(tree));                                        \
            }                                                                                \
            right_rotation_macro(tree);                                                      \
        } else if (weight(left_tree(tree)) < weighted_alpha) {                               \
            if (weight(right_tree(right_tree(tree))) <= ((ALPHA + EPSILON) * weight(tree))) {\
                right_rotation_macro(right_tree(tree));                                      \
            }                                                                                \
            left_rotation_macro(tree);                                                       \
        }                                                                                    \
    }                                                                                        \
})

#define init_tree(tree, _key) ({    \
    set_key(tree, _key);            \
    set_right_tree(tree, NULL);     \
    set_weight(tree, 1);            \
})


void insert(tree_type *tree, unsigned int _key) {
    if (unlikely(is_empty_tree(tree))) {
        init_tree(tree, _key);
        set_left_tree(tree, (void *)-1);
        return ;
    }

    while (is_non_leaf(tree)) {  // record our path down to a leaf ...
        push_tree(tree);
        tree = (_key < key(tree)) ? left_tree(tree) : right_tree(tree);
    }

    tree_type *left_node = allocate_tree(), *right_node = allocate_tree(); // add both leafs ...
    if (_key < key(tree)) {// new key is less than current leaf, move new to the left and previous key to the right ...
        init_tree(left_node, _key);
        init_tree(right_node, key(tree));
    } else {
        init_tree(left_node, key(tree));
        init_tree(right_node, _key);
        set_key(tree, _key);
    }

    set_left_tree(tree, left_node);
    set_right_tree(tree, right_node);
    update_weight(tree);

    balance_weight_tree(tree);
}

// add a node to the tree and cnt the number of entries that are greater than it ...
unsigned int permutation_vector_memb(tree_type *tree, unsigned int _key) {
    if (unlikely(is_empty_tree(tree))) {
        init_tree(tree, _key), set_left_tree(tree, (void *)-1);
        return 0;
    }

    unsigned int count_values_greater = 0;
    while (is_non_leaf(tree)) { // record our path down to a leaf ...
        push_tree(tree);
        if (_key <= key(tree)) // sum up all the previous values that where greater than it ...
            count_values_greater += weight(right_tree(tree));
        tree = (_key < key(tree)) ? left_tree(tree) : right_tree(tree);
    }

    tree_type *left_node = allocate_tree(), *right_node = allocate_tree(); // add both leafs ...
    if (_key < key(tree)) {// new key is less than current leaf, move new to the left and previous key to the right ...
        init_tree(left_node, _key);
        init_tree(right_node, key(tree));
        ++count_values_greater;
    } else {
        init_tree(left_node, key(tree));
        init_tree(right_node, _key), set_key(tree, _key);
    }

    set_left_tree(tree, left_node);
    set_right_tree(tree, right_node);
    update_weight(tree);

    balance_weight_tree(tree);

    return count_values_greater;
}

#define permutation_inversion_ms permutation_inversion // permutation_inversion_ms seems to be the fastest method ...

// n log n solution: gets the inversion sequence of a giving permutation,
//  where a permutation is made up of values from [0, perm_length - 1] without repetitions.
//  an inversion sequence with indices with value a[j] from 0 to n - 1, where a[j] is sum(permutation[0:index_of(j)] > j)
//  basically we are counting all the values preceding j that are greater than j.
void permutation_inversion_wt(perm_elem_type *perm, perm_elem_type *dest, unsigned int perm_length) {
    tree_type *tree = allocate_tree_macro();
    set_left_tree(tree, NULL);

    unsigned word_t index;
    for (index = 0; index < perm_length; index++) // cnt greater elements preceding elemnt at i
        dest[perm[index]] = permutation_vector_memb(tree, perm[index]);

    destroy_tree(tree);
}

// n log n solution: permutation inversion through modified version of merge sort ...
// split the permutation in half apply recursively on each half, until a single element is reached,
// then return (assuming the cnts have already being initialized to zeros ...)
// when returning both halfs have being sorted in ascending order
// when merging every time we take a righ element all the remaining left elements are both bigger and prededing it
// so increment the count for this entry ...
void _permutation_inversion_ms(perm_elem_type *perm, perm_elem_type *dest, unsigned int perm_length) {
    if (perm_length < 2)
        return ;

    typeof(perm_length) left_cnt = half(perm_length), right_cnt = perm_length - left_cnt;
    typeof(perm) left, right;
    _permutation_inversion_ms((left = perm),             dest, left_cnt);
    _permutation_inversion_ms((right = &perm[left_cnt]), dest, right_cnt);

    // merge
    perm_elem_type buffer[perm_length];
    unsigned word_t index;
    for (index = 0; left_cnt && right_cnt; index++)
        if (*left < *right) {
            buffer[index] = *left++;
            left_cnt--;
        } else {  // right is smaller => implies that all the remaining left entries are greater than it
            buffer[index] = *right++;
            right_cnt--;
            dest[buffer[index]] += left_cnt;
        }

    if (left_cnt) // copy any remaining (either) left or right values ...
        memcpy(&buffer[index], left, left_cnt * sizeof(perm[0]));
    else
        memcpy(&buffer[index], right, right_cnt * sizeof(perm[0]));

    memcpy(perm, buffer, perm_length * sizeof(perm[0]));
}

void permutation_inversion_ms(perm_elem_type *perm, perm_elem_type *dest, unsigned int perm_length) {
    typeof(perm[0]) buffer[perm_length];
    _permutation_inversion_ms(
        memcpy(buffer, perm, sizeof(buffer)),
        memset(dest, 0, sizeof(buffer)),
        perm_length
    );
}


// n^2 solution: count the number entries that precreded elemnt i and are greater than i
void permutation_inversion_naive(perm_elem_type *perm, perm_elem_type *dest, unsigned int perm_length) {
    memset(dest, 0, sizeof(*dest) * perm_length); // initialized all counts to zero.

    while (perm_length--){ // count greater preceded values ...
        perm_elem_type *current;
        for (current = &perm[perm_length]; current >= perm; current--)
            dest[perm[perm_length]] += (*current > perm[perm_length]);
    }
}

// returns key containing _weight keys that are currently less than it ...
// ie: to get index i, query the tree on weight, if i is less than the weight of the left tree go to the left
// otherwise subtract the weight of the left from i and go to the right
// once at a leaf remove it, re-balance and return the key of the just removed node ...
unsigned int inv_permutation_vector_memb(tree_type *tree, unsigned int _weight) {
    while (is_non_leaf(tree)) {
        push_tree(tree);
        if (_weight >= weight(left_tree(tree))) {
            _weight -= weight(left_tree(tree));
            tree = right_tree(tree);
        } else
            tree = left_tree(tree);
    }

    typeof(key(tree)) _key = key(tree);

    if (tree_stack_is_non_empty()) {
        tree_type
            *parent = pop_tree(),
            *other_tree = ((left_tree(parent) == tree) ? right_tree(parent) : left_tree(parent));

        *parent = *other_tree;
        recycle_tree(tree);
        recycle_tree(other_tree);

        balance_weight_tree(tree);
    } else
        set_left_tree(tree, NULL);

    return _key;
}

#define perm_from_inv_seq_wt perm_from_inv_seq

// gets a permutation from its inverse sequence where each entry i in the inversion sequence
// is the number of empty places to left of i.
void perm_from_inv_seq_wt(perm_elem_type *inv_seq, perm_elem_type *dest, unsigned int perm_length) {
//    if (__builtin_expect(perm_length == 0 || is_null(inv_seq) || is_null(dest), 0))
//        return ;
//    if (__builtin_expect(perm_length == 1, 0)) {
//        *dest = 0;
//        return ;
//    }

    unsigned word_t index = 0;
    // build a weight balance binary tree containing 0 .. perm_length - 1 keys
    // top down construction: select the middle element, build left and right tree and join them
    // left tree contains elements from [0 ... n / 2) right tree contains [n/2 ... n) with root at n/2
    struct {typeof(perm_length) start; tree_type *tree;} stack[100];
    tree_type *tree = new_tree(half(perm_length), perm_length), *_parent;
    stack[index++] = (typeof(stack[0])){.start = 0, .tree = tree};

    while (index)
        if (weight((_parent = stack[--index].tree)) > 1) {
            typeof(perm_length) start = stack[index].start, length = weight(_parent);
            typeof(tree)
                left_t = new_tree(start + quarter(length), half(length)),
                right_t = allocate_tree();

            set_weight(right_t, length - half(length));
            set_key(right_t, key(_parent) + half(weight(right_t)));
            set_right_tree(right_t, NULL);

            set_left_tree(_parent, left_t);
            set_right_tree(_parent, right_t);

            stack[index++] = (typeof(stack[0])){.start = key(_parent),  .tree = right_t};
            stack[index++] = (typeof(stack[0])){.start = start,         .tree = left_t};
        }
//    tree_type *tree = allocate_tree();
//    set_left_tree(tree, NULL);
//    for (index = 0; index < perm_length; index++)
//        insert(tree, index);

    for (index = 0; index < perm_length; index++)
        dest[inv_permutation_vector_memb(tree, inv_seq[index])] = index;

    recycle_tree(tree);
}

// get permutation from inversion vector using merge sort ...
// create a list of values from [0 ... n-1]
// split the list in half, apply recursively until a single element is reached in which case simply return ...
// when returning we want both halfs sorted in ascending order, when we merge
void _perm_from_inv_seq_ms(perm_elem_type *weights, perm_elem_type *dest, unsigned int perm_length) {
    if (perm_length < 2)
        return ;
//
//    typeof(perm_length) left_cnt = half(perm_length), right_cnt = perm_length - left_cnt;
//    typeof(perm) left, right;
//    _permutation_inversion_ms((left = weights),             dest, left_cnt);
//    _permutation_inversion_ms((right = &perm[left_cnt]), dest, right_cnt);

}

void perm_from_inv_seq_ms(perm_elem_type *weights, perm_elem_type *dest, unsigned int perm_length) {
    unsigned word_t index;
    for (index = 0; index < perm_length; index++)
        *dest = index;
    typeof(weights[0]) buffer[perm_length];
    _perm_from_inv_seq_ms(memcpy(buffer, weights, sizeof(buffer)), dest, perm_length);
}



void random_perm(perm_elem_type *dest, unsigned cnt) {
    unsigned index;
    for (index = 0; index < cnt; index++) { // randomly swap two positions ...
        typeof(index) swap_index = rand() % (index + 1); // pick a previously inserted value ...
        dest[index] = dest[swap_index]; // swap it with current ...
        dest[swap_index] = index;
    }
}

#define timeit_repeat(func, n) ({ unsigned word_t index; for (index = 0; index < (n); index++) func(); })

int main() {
//    perm_elem_type values[] = {
//        18, 15  ,1  ,6  ,5 ,14 ,13 ,16  ,4  ,7 ,
//        8  ,0 ,17 ,11 ,10 ,19  ,2  ,3  ,9 ,12
////    3, 0, 4 ,1 ,5 ,2
//    //    5,4,3,2,1,0
//    };

    perm_elem_type count = 128;
    perm_elem_type values[count];
    random_perm(values, count);
//    int index;

    perm_elem_type inv_perm[count];

    #define REPEAT_CNT 1

    #define perm_inv_naive() ({permutation_inversion_naive(values, inv_perm, count); })
    #define timed_perm_inv_naive() timeit_repeat(perm_inv_naive, REPEAT_CNT)
    printf("permutation_inversion_naive: %.4fs \n", timed(timed_perm_inv_naive));
//    permutation_inversion_naive(values, inv_perm, count);
//    for (index = 0; index < count; index++)
//        printf("%u ", inv_perm[index]);
//    printf("\n");

    typeof(inv_perm) inv_perm_wt;
    #define perm_inv_tree() ({permutation_inversion_wt(values, inv_perm_wt, count);})
    #define timed_perm_inv_tree() timeit_repeat(perm_inv_tree, REPEAT_CNT)
    printf("permutation_inversion_weight_tree: %.4fs ", timed(timed_perm_inv_tree));
    printf( (memcmp(inv_perm_wt, inv_perm, sizeof(inv_perm))) ? "failed!! \n": "ok. \n");


    typeof(inv_perm) inv_perm_ms;
    #define perm_inv_ms() ({permutation_inversion_ms(values, inv_perm_ms, count);})
    #define timed_perm_inv_ms() timeit_repeat(perm_inv_ms, REPEAT_CNT)
    printf("permutation_inversion_merge_sort: %.4fs ", timed(timed_perm_inv_ms));
    printf( (memcmp(inv_perm_ms, inv_perm, sizeof(inv_perm))) ? "failed!! \n" : "ok. \n");
//    permutation_inversion_ms(values, inv_perm, count);
//    for (index = 0; index < count; index++)
//        printf("%u ", inv_perm[index]);
//    printf("\n");

//    permutation_inversion_wt(values, inv_perm, count);
//    for (index = 0; index < count; index++)
//        printf("%u ", inv_perm[index]);
//    printf("\n");


    typeof(values) temp;
    perm_from_inv_seq_wt(inv_perm, temp, count);
    printf("perm_from_inv_seq_weight_tree %s \n", (memcmp(temp, values, sizeof(values))) ? "failed!!" : "ok.");



    return 0;
}