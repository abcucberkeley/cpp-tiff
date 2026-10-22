#include "lzwencode.h"
#include <cstdint>
#include <cstdlib>
#include <assert.h>
#include <mutex>
#include <vector>
#include "tiffio.h"
typedef size_t WordType;

/*
 * NB: The 5.0 spec describes a different algorithm than Aldus
 *     implements.  Specifically, Aldus does code length transitions
 *     one code earlier than should be done (for real LZW).
 *     Earlier versions of this library implemented the correct
 *     LZW algorithm, but emitted codes in a bit order opposite
 *     to the TIFF spec.  Thus, to maintain compatibility w/ Aldus
 *     we interpret MSB-LSB ordered codes to be images written w/
 *     old versions of this library, but otherwise adhere to the
 *     Aldus "off by one" algorithm.
 *
 * Future revisions to the TIFF spec are expected to "clarify this issue".
 */

#define MAXCODE(n)	((1L<<(n))-1)
/*
 * The TIFF spec specifies that encoded bit
 * strings range from 9 to 12 bits.
 */
#define BITS_MIN        9               /* start with 9 bits */
#define BITS_MAX        12              /* max of 12 bit strings */
/* predefined codes */
#define CODE_CLEAR      256             /* code to clear string table */
#define CODE_EOI        257             /* end-of-information code */
#define CODE_FIRST      258             /* first free code entry */
#define CODE_MAX        MAXCODE(BITS_MAX)

#define CHECK_GAP	10000		/* enc_ratio check interval */

#define	CALCRATIO(sp, rat) {					\
    if (incount > 0x007fffff) { /* NB: shift will overflow */\
    rat = outcount >> 8;				\
    rat = (rat == 0 ? 0x7fffffff : incount/rat);	\
    } else							\
    rat = (incount<<8) / outcount;			\
    }

/* Explicit 0xff masking to make icc -check=conversions happy */
#define	PutNextCode(op, c) {					\
    nextdata = (nextdata << nbits) | c;			\
    nextbits += nbits;					\
    *op++ = (unsigned char)((nextdata >> (nextbits-8))&0xff);		\
    nextbits -= 8;						\
    if (nextbits >= 8) {					\
    *op++ = (unsigned char)((nextdata >> (nextbits-8))&0xff);	\
    nextbits -= 8;					\
    }							\
    outcount += nbits;					\
    }

namespace {

// The string table. child[c << 12 | ent] is the code of string ent followed by
// byte c, or 0 if that string is not in the table (codes below 258 are never
// added, so 0 is free). libtiff finds strings through a 9001-entry hash table
// instead; indexing directly finds the same strings, so the codes and the
// output are the same, but encodes 1.3-3x faster. The table is laid out by
// byte first because image data repeats few byte values in some positions
// (the high bytes of 16-bit samples), which keeps those lookups close together.
// A reset empties only the entries added, so the 2 MB table is never cleared
// as a whole.
struct LzwTable {
    uint16_t child[256 << BITS_MAX];
    uint32_t added[1 << BITS_MAX];
};

// Tables kept between encodes, so a strip does not allocate and fault in 2 MB.
// There are as many as the most strips ever encoded at once.
struct TablePool {
    std::mutex m;
    std::vector<LzwTable*> tables;
    ~TablePool(){
        for(LzwTable* t : tables) free(t);
    }
};
TablePool pool;

LzwTable* takeTable(){
    {
        std::lock_guard<std::mutex> g(pool.m);
        if(!pool.tables.empty()){
            LzwTable* t = pool.tables.back();
            pool.tables.pop_back();
            return t;
        }
    }
    return (LzwTable*)calloc(1, sizeof(LzwTable));
}

void returnTable(LzwTable* t){
    std::lock_guard<std::mutex> g(pool.m);
    pool.tables.push_back(t);
}

}

// Encodes len bytes as one LZW strip (libtiff's encoder: the same codes, code
// widths and clear codes, so the same bytes). Returns the bytes written to
// compr, or 0 if the string table could not be allocated.
uint64_t lzwEncode(uint8_t* unCompr, uint8_t* compr, tmsize_t cc){
    LzwTable* table = takeTable();
    if(!table) return 0;
    uint16_t* child = table->child;
    uint32_t* added = table->added;
    int nAdded = 0;
#define RESET_TABLE { for(int i_ = 0; i_ < nAdded; i_++) child[added[i_]] = 0; nAdded = 0; }

    uint8_t* bp = unCompr;
    uint8_t* op = compr;
    tmsize_t incount = 0, outcount = 0, checkpoint = CHECK_GAP, enc_ratio = 0;
    WordType nextdata = 0;
    long nextbits = 0;
    int free_ent = CODE_FIRST, maxcode = MAXCODE(BITS_MIN), nbits = BITS_MIN;
    int ent = -1;

    if (cc > 0) {
        PutNextCode(op, CODE_CLEAR);
        ent = *bp++; cc--; incount++;
    }
    while (cc > 0) {
        const uint32_t c = *bp++; cc--; incount++;
        const uint32_t idx = (c << BITS_MAX) | (uint32_t)ent;
        const uint16_t code = child[idx];
        if (code) {
            ent = code;
            continue;
        }
        /*
         * New entry, emit code and add to table.
         */
        PutNextCode(op, ent);
        ent = (int)c;
        child[idx] = (uint16_t)(free_ent++);
        added[nAdded++] = idx;
        if (free_ent == CODE_MAX-1) {
            /* table is full, emit clear code and reset */
            RESET_TABLE
            enc_ratio = 0;
            incount = 0;
            outcount = 0;
            free_ent = CODE_FIRST;
            PutNextCode(op, CODE_CLEAR);
            nbits = BITS_MIN;
            maxcode = MAXCODE(BITS_MIN);
        } else {
            /*
             * If the next entry is going to be too big for
             * the code size, then increase it, if possible.
             */
            if (free_ent > maxcode) {
                nbits++;
                assert(nbits <= BITS_MAX);
                maxcode = (int) MAXCODE(nbits);
            } else if (incount >= checkpoint) {
                tmsize_t rat;
                /*
                 * Check compression ratio and, if things seem
                 * to be slipping, clear the hash table and
                 * reset state.  The compression ratio is a
                 * 24+8-bit fractional number.
                 */
                checkpoint = incount+CHECK_GAP;
                CALCRATIO(sp, rat);
                if (rat <= enc_ratio) {
                    RESET_TABLE
                    enc_ratio = 0;
                    incount = 0;
                    outcount = 0;
                    free_ent = CODE_FIRST;
                    PutNextCode(op, CODE_CLEAR);
                    nbits = BITS_MIN;
                    maxcode = MAXCODE(BITS_MIN);
                } else
                    enc_ratio = rat;
            }
        }
    }
    RESET_TABLE
#undef RESET_TABLE
    returnTable(table);

    if (ent != -1) {
        // Flush the final pending code.
        PutNextCode(op, ent);
        ent = -1;
        free_ent ++;

        if (free_ent == CODE_MAX-1) {
            /* table is full, emit clear code and reset */
            PutNextCode(op, CODE_CLEAR);
            nbits = BITS_MIN;
        } else {
            /*
             * If the next entry is going to be too big for
             * the code size, then increase it, if possible.
             */
            if (free_ent > maxcode) {
                nbits++;
                assert(nbits <= BITS_MAX);
            }
        }
    }
    PutNextCode(op, CODE_EOI);
    /* Explicit 0xff masking to make icc -check=conversions happy */
    if (nextbits > 0){
        *op++ = (unsigned char)((nextdata << (8-nextbits))&0xff);
        outcount += nbits;
    }
    (void)outcount;
    // The bytes actually written. Rounding the bit count up instead could count
    // a byte or two of the uninitialized buffer past the last code, so the same
    // data did not always produce the same file.
    return (uint64_t)(op - compr);
}
