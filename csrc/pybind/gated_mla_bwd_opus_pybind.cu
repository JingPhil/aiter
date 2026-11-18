// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/gated_mla_bwd_opus.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    GATED_MLA_BWD_OPUS_PYBIND;
}
