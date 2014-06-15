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
    #error "Unable to determine machine word type"
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

#define swap_xchg(a, b) ({                              \
    switch(sizeof(a)) {                                 \
        case 1: instr(swap_xchg_, 1)(a, b); break ;     \
        case 2: instr(swap_xchg_, 2)(a, b); break ;     \
        case 4: instr(swap_xchg_, 4)(a, b); break ;     \
        case 8: instr(swap_xchg_, 8)(a, b); break ;     \
    }                                                   \
})

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

#define half(a)     rshift_logcl(a, 1)
#define quarter(a)  rshift_logcl(a, 2)

#define unlikely(expr) __builtin_expect(expr, 0)
#define likely(expr) __builtin_expect(expr, 1)

#define is_non_null(v) (v)
#define is_null(v) (!is_non_null(v))


#define timed(func) ({                              \
    clock_t __start__, __end__;                     \
    __start__ = clock();                            \
        func();                                     \
    __end__ = clock();                              \
    ((double)(__end__ - __start__))/CLOCKS_PER_SEC; \
})


#define perm_elem_type unsigned char

// diaganol collision occur between two points when there |delta(x cords)| == |delta(y cords)|
// ie the mag in their corresponding x and y coordinates equal
// and theres no other queen in between them ....
// there can never be more than n - 1 diagonal collision in any giving board of size n with n queens ...
// assuming there are no row or column collisions (col_cords has unique values)
// there are a total of 4n diagnols in each board of size n
// a queen may either be in (y cord - x cord) diag and/or the ((n - y cord) - x cord) diag respectively
unsigned int diag_collisions(perm_elem_type *col_cords, unsigned int board_size) {
    word_t
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

    for (index = 0; index < sizeof(cnts)/sizeof(cnts[0]); index++) // count collisions number of queens in max(diag, 1) - 1
        if (cnts[index])
            sum += cnts[index] - 1;
//        sum += max(cnts[index], 1) - 1;

    return sum;
}



// weight balanced tree ...
#define TREE_BLOCK_MAG 128
#define ALPHA    0.22 //0.288
#define EPSILON  0.005
typedef struct tree_type {
    struct tree_type
        *_left,
        *_right;

    unsigned _key, _weight;
} tree_type;

#define left_tree(t) (t->_left)
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
word_t available_trees = TREE_BLOCK_MAG;
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

#define init_tree(tree, _key, _weight) ({       \
    set_key(tree, _key);                        \
    set_right_tree(tree, NULL);                 \
    set_weight(tree, _weight);                  \
tree; })

#define init_empty_tree(tree, _key, _value) ({  \
    set_key(tree, _key);                        \
    set_weight(tree, 1);                        \
    set_right_tree(tree, NULL);                 \
    set_left_tree(tree, _value);                \
tree; })

#define new_tree(key, weight) ({ tree_type *_n_tree_ = allocate_tree_macro();  init_tree(_n_tree_, key, weight); _n_tree_;})
#define new_leaf(key) new_tree(key, 1)

#define new_empty_tree() ({ tree_type *tree = allocate_tree(); set_left_tree(tree, NULL); tree; })

#define update_weight(tree) set_weight(tree, ((weight(left_tree(tree)) + weight(right_tree(tree)))))


#define left_rotation_macro(tree) ({                                \
    typeof(tree) r_tree = right_tree(tree);                         \
    swap(key(tree),                 key(r_tree));                   \
    swap(left_tree(tree),           right_tree(tree));              \
    swap(right_tree(r_tree),        left_tree(r_tree));             \
    swap(right_tree(tree),          left_tree(left_tree(tree)));    \
    update_weight(r_tree);                                          \
})


#define right_rotation_macro(tree) ({                           \
    typeof(tree) l_tree = left_tree(tree);                      \
    swap(key(tree),             key(l_tree));                   \
    swap(left_tree(tree),       right_tree(tree));              \
    swap(left_tree(l_tree),     right_tree(l_tree));            \
    swap(left_tree(tree),       right_tree(right_tree(tree)));  \
    update_weight(l_tree);                                      \
})

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


void insert(tree_type *tree, unsigned int _key) {
    if (unlikely(is_empty_tree(tree))) {
        init_empty_tree(tree, _key, (void *)-1);
        return ;
    }

    while (is_non_leaf(tree)) {  // record our path down to a leaf ...
        push_tree(tree);
        tree = (_key < key(tree)) ? left_tree(tree) : right_tree(tree);
    }

    tree_type *node_a = new_leaf(_key), *node_b = new_leaf(key(tree)); // add both leafs ...
    if (_key < key(tree)) {// new key is less than current leaf, move new to the left and previous key to the right ...
        set_left_tree(tree, node_a);
        set_right_tree(tree, node_b);
    } else {
        set_left_tree(tree, node_b);
        set_right_tree(tree, node_a);
        set_key(tree, _key);
    }

    update_weight(tree);

    balance_weight_tree(tree);
}



#define permutation_inversion_ms    permutation_inversion // permutation_inversion_ms seems to be the fastest method ...
#define perm_from_inv_seq_ms        perm_from_inv_seq           // again seems to be the fastest ...



// add a node to the tree and cnt the number of entries that are greater than it ...
unsigned int permutation_vector_memb(tree_type *tree, unsigned int _key) {
    if (unlikely(is_empty_tree(tree))) {
        init_empty_tree(tree, _key, (void *)-1);
        return 0;
    }

    unsigned int count_values_greater = 0;
    while (is_non_leaf(tree)) { // record our path down to a leaf ...
        push_tree(tree);
        if (_key <= key(tree)) // sum up all the previous values that where greater than it ...
            count_values_greater += weight(right_tree(tree));
        tree = (_key < key(tree)) ? left_tree(tree) : right_tree(tree);
    }

    tree_type *leaf_a = new_leaf(_key), *leaf_b = new_leaf(key(tree)); // add both leafs ...
    if (_key < key(tree)) {// new key is less than current leaf, move new to the left and previous key to the right ...
        set_left_tree(tree, leaf_a);
        set_right_tree(tree, leaf_b);
        ++count_values_greater;
    } else {
        set_left_tree(tree, leaf_b);
        set_right_tree(tree, leaf_a);
        set_key(tree, _key);
    }

    update_weight(tree);

    balance_weight_tree(tree);

    return count_values_greater;
}

// n log n solution: gets the inversion sequence of a giving permutation,
//  where a permutation is made up of values from [0, perm_length - 1] without repetitions.
//  an inversion sequence with indices with value a[j] from 0 to n - 1, where a[j] is sum(permutation[0:index_of(j)] > j)
//  basically we are counting all the values preceding j that are greater than j.
void permutation_inversion_wt(perm_elem_type *perm, perm_elem_type *dest, unsigned int perm_length) {
    tree_type *tree = new_empty_tree();

    unsigned word_t index;
    for (index = 0; index < perm_length; index++) // cnt greater elements preceding elemnt at i
        dest[perm[index]] = permutation_vector_memb(tree, perm[index]);

    destroy_tree(tree);
}

// n log n solution: permutation inversion through modified version of merge sort ...
// split the permutation in half apply recursively on each half, until a single element is reached,
// then return (assuming the cnts have already being initialized to zeros ...)
// when returning both halfs have being sorted in ascending order
// when merging every time we take a righ element all the remaining left elements are both bigger and preceeding it
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
void permutation_inversion_naive(perm_elem_type *perm, perm_elem_type *dest, unsigned int cnt) {
    memset(dest, 0, sizeof(*dest) * cnt); // initialized all counts to zero.

    while (cnt--){ // count greater preceded values ...
        perm_elem_type *current;
        for (current = &perm[cnt]; current >= perm; current--)
            dest[perm[cnt]] += (*current > perm[cnt]);
//            if (*current > perm[cnt])
//                dest[perm[cnt]]++;
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

// gets a permutation from its inverse sequence where each entry i in the inversion sequence
// is the number of empty places to left of i.
void perm_from_inv_seq_wt(perm_elem_type *inv_seq, perm_elem_type *dest, unsigned int perm_length) {

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

    for (index = 0; index < perm_length; index++)
        dest[inv_permutation_vector_memb(tree, inv_seq[index])] = index;

    recycle_tree(tree);
}

// naive approach (n^2): initilize dest to inv_vect,
// iterate backwards over dest at each index i, check all the elements starting at i + 1
// are greater than or equal to dest[i], if they are increment those that are otherwise continue
// in essence we are shifting conflicting positions ...
// once complete argsort dest

// divide & conquer (n log n):
// assuming indices have being initialized with the identity permutation [0 ... n - 1]
// divide the indices in half and apply recursively, until a single element is reached then nothing to do so just return.
// upon retuning we can assume that:
// 1) the indices, have being sorted in ascending order based on the offsets.
// 2) offsets have being shifted so as to take into account all the preceding offsets, that are smaller than it.
// In order to maintain both this properties we:
// when we merge, we iterate over each half in ascending order using the indices
// if the current left magnitude is less than the current right_magnitude shifted than
// nothing to do take left and continue.
// otherwise left magnitude is greater than right shifted (ie it supercedes it) so update right and take right index ...
void _perm_from_inv_seq_ms(perm_elem_type *offsets, perm_elem_type *indices, unsigned int cnt) {
    if (cnt < 2)
        return ;

    typeof(cnt) left_cnt = half(cnt), right_cnt = cnt - left_cnt;
    typeof(indices) left_indices = indices, right_indices = &indices[left_cnt];

    _perm_from_inv_seq_ms(offsets,  left_indices,  left_cnt);
    _perm_from_inv_seq_ms(offsets, right_indices, right_cnt);

    typeof(indices[0]) argsort[cnt];
    unsigned word_t index, left_index = 0;
    for (index = 0; (left_index < left_cnt) && right_cnt; index++) {
        if (offsets[left_indices[left_index]] <= (offsets[*right_indices] + left_index))
            argsort[index] = left_indices[left_index++]; // take left entry ...
        else {
            argsort[index] = *right_indices;
            offsets[*right_indices++] += left_index; // update right to account for all the entries that are smaller than it.
            right_cnt--;
        }
    }

    if (right_cnt) { // copy and update any remaining entries ...
        memcpy(&argsort[index], right_indices, right_cnt * sizeof(right_indices[0]));
        for (index = 0; index < right_cnt; index++)
            offsets[right_indices[index]] += left_index; // left_cnt == left_index
    } else
        memcpy(&argsort[index], &left_indices[left_index], (left_cnt - left_index) * sizeof(left_indices[0]));

    memcpy(indices, argsort, sizeof(argsort));
}

void perm_from_inv_seq_ms(perm_elem_type *inversion_vect, perm_elem_type *perm, unsigned int cnt) {
    typeof(inversion_vect[0]) buffer[cnt];
    unsigned word_t index;
    for (index = 0; index < cnt; index++)
        perm[index] = index;

    _perm_from_inv_seq_ms(
        memcpy(buffer, inversion_vect, sizeof(buffer)), perm, cnt
    );
}

// generate permutation perm based on its inversion vector.
// naive approach (n^2) simply count the number of empty slots to the left
void perm_from_inv_seq_naive(perm_elem_type *inversion_vect, perm_elem_type *perm, unsigned int cnt) {
    memset(perm, -1, sizeof(perm[0]) * cnt);

    unsigned word_t index;
    for (index = 0; index < cnt; index++) {
        typeof(inversion_vect[0]) empty_cnt = inversion_vect[index];
        typeof(index) loc;

        for (loc = 0; empty_cnt || (perm[loc] != (typeof(inversion_vect[0]))-1); loc++) // while we have consumed enough empty slots or we are not in empty spot ...
            if (empty_cnt && perm[loc] == (typeof(inversion_vect[0]))-1) // empty spot
                empty_cnt--;

//        if (likely(loc < cnt))
            perm[loc] = index;
    }
}


void random_perm(perm_elem_type *dest, unsigned cnt) { // knuth shuffle ...
    unsigned word_t index;
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
//        5,4,3,2,1,0
//    };

//    unsigned count = sizeof(values)/sizeof(values[0]);
//    perm_elem_type values[] = {4, 1, 5, 2, 0, 3};

    unsigned count = 100000;
    perm_elem_type values[count];
    random_perm(values, count);

    perm_elem_type inv_perm[count];

    #define perm_inv_naive() ({permutation_inversion_naive(values, inv_perm, count); })
    printf("permutation_inversion_naive: %.4fs \n", timed(perm_inv_naive));

    typeof(inv_perm) inv_perm_wt;
    #define perm_inv_wt() ({permutation_inversion_wt(values, inv_perm_wt, count);})
    printf("permutation_inversion_wt: %.4fs ", timed(perm_inv_wt));
    printf((memcmp(inv_perm_wt, inv_perm, sizeof(inv_perm))) ? "failed!! \n": "ok. \n");


    typeof(inv_perm) inv_perm_ms;
    #define perm_inv_ms() ({permutation_inversion_ms(values, inv_perm_ms, count);})
    printf("permutation_inversion_ms: %.4fs ", timed(perm_inv_ms));
    printf((memcmp(inv_perm_ms, inv_perm, sizeof(inv_perm))) ? "failed!! \n" : "ok. \n");

    printf("\n");

    typeof(values) values_ms;
    #define inv_ms() ({ perm_from_inv_seq_ms(inv_perm, values_ms, count); })
    printf("perm_from_inv_seq_ms %.4fs ", timed(inv_ms));
    printf((memcmp(values_ms, values, sizeof(values))) ? "failed!!\n" : "ok.\n");

    typeof(values) values_wt;
    #define inv_wt() ({ perm_from_inv_seq_wt(inv_perm, values_wt, count); })
    printf("perm_from_inv_seq_weight_tree %.4fs ", timed(inv_wt));
    printf((memcmp(values_wt, values, sizeof(values))) ? "failed!!\n" : "ok.\n");


    typeof(values) values_naive;
    #define inv_naive() ({ perm_from_inv_seq_naive(inv_perm, values_naive, count); })
    printf("perm_from_inv_seq_naive: %.4fs ", timed(inv_naive));
    printf((memcmp(values_naive, values, sizeof(values))) ? "failed!!\n" : "ok.\n");

    return 0;
}