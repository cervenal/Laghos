// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_timeinteg.hpp"
#include "laghos_solver.hpp"

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void HydroODESolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);

   hydro_oper = dynamic_cast<LagrangianHydroOperator *>(f);
   MFEM_VERIFY(hydro_oper, "HydroSolvers expect LagrangianHydroOperator.");
}

void HydroODESolver::GenericStep(Vector &S, double &t, double &dt,
                                 int stages, const double b[], const double bbar[],
                                 const double A[][10])
{
   const int Vsize = hydro_oper->GetH1VSize();
   Vector V(Vsize), S0(S);

   Vector *dS_dt = new Vector[stages];
   Vector *dv_dt = new Vector[stages];
   for (int i = 0; i < stages; i++)
   {
      dS_dt[i].SetSize(S.Size());
      dv_dt[i].SetDataAndSize(dS_dt[i].GetData() + Vsize, Vsize);
   }

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector v0, dx_dt;
   v0.SetDataAndSize(S0.GetData() + Vsize, Vsize);
   V = v0;

   // In each sub-step:
   // - Update the global state Vector S.
   // - Compute dv_dt using S.
   // - Update V using dv_dt.
   // - Compute de_dt and dx_dt using S and V.
   for (int i = 0; i < stages; i++)
   {
      // S_i = S0 + sum_j dt A_ij dS_dt_j.
      S = S0;
      for (int j = 0; j < i; j++) { S.Add(A[i][j] * dt, dS_dt[j]); }

      // Compute dv_dt_i using S_i
      hydro_oper->ResetQuadratureData();
      hydro_oper->UpdateMesh(S);
      hydro_oper->SolveVelocity(S, dS_dt[i]);

      // Update V.
      if (i > 0) { V.Add(0.5 * b[i-1] * dt, dv_dt[i-1]); }
      V.Add(0.5 * b[i] * dt, dv_dt[i]);

      // Compute de_dt_i and dx_dt_i using S_i and V.
      hydro_oper->SolveEnergy(S, V, dS_dt[i]);
      dx_dt.SetDataAndSize(dS_dt[i].GetData(), Vsize);
      dx_dt = V;
   }

   S = S0;
   for (int i = 0; i < stages; i++) { S.Add(b[i] * dt, dS_dt[i]); }
   //for (int i = 0; i < stages; i++) { S.Add(bbar[i] * dt, dS_dt[i]); }
   hydro_oper->ResetQuadratureData();

   t += dt;

   delete [] dv_dt;
   delete [] dS_dt;
}

void RK2AvgSolver::Step(Vector &S, double &t, double &dt)
{
   const int Vsize = hydro_oper->GetH1VSize();
   Vector V(Vsize), dS_dt(S.Size()), S0(S);

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector dv_dt, v0, dx_dt;
   v0.SetDataAndSize(S0.GetData() + Vsize, Vsize);
   dv_dt.SetDataAndSize(dS_dt.GetData() + Vsize, Vsize);
   dx_dt.SetDataAndSize(dS_dt.GetData(), Vsize);

   // In each sub-step:
   // - Update the global state Vector S.
   // - Compute dv_dt using S.
   // - Update V using dv_dt.
   // - Compute de_dt and dx_dt using S and V.

   // -- 1.
   // S is S0.
   hydro_oper->UpdateMesh(S);
   hydro_oper->SolveVelocity(S, dS_dt);
   // V = v0 + 0.5 * dt * dv_dt;
   add(v0, 0.5 * dt, dv_dt, V);
   hydro_oper->SolveEnergy(S, V, dS_dt);
   dx_dt = V;

   // -- 2.
   // S = S0 + 0.5 * dt * dS_dt;
   add(S0, 0.5 * dt, dS_dt, S);
   hydro_oper->ResetQuadratureData();
   hydro_oper->UpdateMesh(S);
   hydro_oper->SolveVelocity(S, dS_dt);
   // V = v0 + 0.5 * dt * dv_dt;
   add(v0, 0.5 * dt, dv_dt, V);
   hydro_oper->SolveEnergy(S, V, dS_dt);
   dx_dt = V;

   // -- 3.
   // S = S0 + dt * dS_dt.
   add(S0, dt, dS_dt, S);
   hydro_oper->ResetQuadratureData();

   t += dt;
}

void RK3hcAalphaSolver::Step(Vector &S, double &t, double &dt)
{
   double b[4]     = {  0.0, 1.14490726777366794411117601160, -1.93000000000000000000000000000,    1.78509273222633205588882398840};
   double A[4][10] = {{ 0.0,                                 0.0,                                0.0,                                  0.0},
                      { 0.572453633886833972055588005800,    0.0,                                0.0,                                  0.0},
                      { 1.73,                               -1.55009273222633205588882398840,    0.0,                                  0.0},
                      { 1.62,                               -1.51295406123109734930854618380,    0.000407695117931321364134189600056,  0.0}};

   GenericStep(S, t, dt, 4, b, b, A);
}

void RK4hcAalphaSolver::Step(Vector &S, double &t, double &dt)
{
   double b[5]     = {0, 1.05624613715978312392837289073, -1.18876511778398048300584071027, 0.768066441284648838442959468355, 0.364452539339548520634508351188};
   double A[5][10] = {{ 0.0,                                    0.0,                                0.0,                                0.0,                               0.0},
                      { 0.528123068579891561964186445364,       0.0,                                0.0,                                0.0,                               0.0},
                      { 0.594382558891990241502920355135,      -0.132518980624197359077467819542,   0.0,                                0.0,                               0.0},
                      { 0.349146376428782396125720350727,       0.0450000000000000000000000000000,  -0.142632136410655335981708436093,  0.0,                               0.0},
                      { -0.167439902559297058870190958083,      1.82882953442616803795893691307,   -2.07222863920254151884340051116,    1.22861273766589627943740038057,   0.0}};

   GenericStep(S, t, dt, 5, b, b, A);
}

} // namespace hydrodynamics

} // namespace mfem
