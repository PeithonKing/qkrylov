#pragma once

#include "qkrylov/core/types.hpp"

#include <string>
#include <vector>

namespace qkrylov
{

struct OperatorFactor
{
    std::string op;
    int site;
};

struct OperatorTerm
{
    Complex coeff;

    std::vector<OperatorFactor> factors;

    OperatorTerm() = default;

    OperatorTerm(Complex c, std::initializer_list<OperatorFactor> f)
        : coeff(c), factors(f) {}
};

// --- Algebraic Expression Template Syntax ---

// Represents a single operator on a site (e.g. Sz(i))
struct LocalOp {
    std::string name;
    int site;
};

// 1.0 * Sz(i)
inline OperatorTerm operator*(Complex coeff, const LocalOp& op) {
    OperatorTerm t;
    t.coeff = coeff;
    t.factors.push_back({op.name, op.site});
    return t;
}

inline OperatorTerm operator*(double coeff, const LocalOp& op) {
    return Complex(coeff, 0.0) * op;
}

// Sz(i) * Sz(i+1) (defaults to coefficient 1.0)
inline OperatorTerm operator*(const LocalOp& op1, const LocalOp& op2) {
    OperatorTerm t;
    t.coeff = 1.0;
    t.factors.push_back({op1.name, op1.site});
    t.factors.push_back({op2.name, op2.site});
    return t;
}

// Term * Sz(i)
inline OperatorTerm operator*(OperatorTerm term, const LocalOp& op) {
    term.factors.push_back({op.name, op.site});
    return term;
}

// Term * Term (groups factors and multiplies coefficients)
inline OperatorTerm operator*(OperatorTerm t1, const OperatorTerm& t2) {
    t1.coeff *= t2.coeff;
    t1.factors.insert(t1.factors.end(), t2.factors.begin(), t2.factors.end());
    return t1;
}

// Expression to hold multiple terms (e.g. Term + Term)
struct OpSumExpr {
    std::vector<OperatorTerm> terms;
    OpSumExpr(const OperatorTerm& t) : terms({t}) {}
    OpSumExpr(const std::vector<OperatorTerm>& ts) : terms(ts) {}
};

inline OpSumExpr operator+(const OperatorTerm& t1, const OperatorTerm& t2) {
    return OpSumExpr({t1, t2});
}

inline OpSumExpr operator+(OpSumExpr expr, const OperatorTerm& t) {
    expr.terms.push_back(t);
    return expr;
}

inline OpSumExpr operator+(const OperatorTerm& t, OpSumExpr expr) {
    expr.terms.insert(expr.terms.begin(), t);
    return expr;
}

inline OpSumExpr operator+(OpSumExpr expr1, const OpSumExpr& expr2) {
    expr1.terms.insert(expr1.terms.end(), expr2.terms.begin(), expr2.terms.end());
    return expr1;
}

inline OpSumExpr operator-(const OperatorTerm& t1, OperatorTerm t2) {
    t2.coeff = -t2.coeff;
    return OpSumExpr({t1, t2});
}

inline OpSumExpr operator-(OpSumExpr expr, OperatorTerm t) {
    t.coeff = -t.coeff;
    expr.terms.push_back(t);
    return expr;
}

} // namespace qkrylov