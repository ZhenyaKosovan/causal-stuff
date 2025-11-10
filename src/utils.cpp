#include "utils.h"

// This translation unit is intentionally empty: including "utils.h" here ensures
// the header compiles on its own, and gives the linker a consistent object file
// even though every helper is currently header-only. Keeping the stub allows us
// to move functionality out of the header in the future without touching the
// build system again.
