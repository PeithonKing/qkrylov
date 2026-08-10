#pragma once

#include "qkrylov/core/types.hpp"

#include "operator_term.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


// Helper functions for common operators to enable algebraic syntax
inline LocalOp Sz(int i) { return {"Sz", i}; }
inline LocalOp Sp(int i) { return {"Sp", i}; }
inline LocalOp Sm(int i) { return {"Sm", i}; }
inline LocalOp Sx(int i) { return {"Sx", i}; }
inline LocalOp Sy(int i) { return {"Sy", i}; }

inline LocalOp CdagUp(int i) { return {"CdagUp", i}; }
inline LocalOp CUp(int i) { return {"CUp", i}; }
inline LocalOp CdagDn(int i) { return {"CdagDn", i}; }
inline LocalOp CDn(int i) { return {"CDn", i}; }

inline LocalOp Nup(int i) { return {"Nup", i}; }
inline LocalOp Ndn(int i) { return {"Ndn", i}; }
inline LocalOp Nupdn(int i) { return {"Nupdn", i}; }

inline LocalOp Bdag(int i) { return {"Bdag", i}; }
inline LocalOp B(int i) { return {"B", i}; }
inline LocalOp N(int i) { return {"N", i}; }

}

}
