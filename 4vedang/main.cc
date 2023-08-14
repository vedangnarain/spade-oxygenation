// #include <chrono>
#include "scidf.h"
#include "spade.h"
#include "vgeom.h"

using real_t = double;
using flux_t = spade::fluid_state::flux_t<real_t>;
using prim_t = spade::fluid_state::prim_t<real_t>;
using cons_t = spade::fluid_state::cons_t<real_t>;

namespace local
{
    template <typename float_t> struct ldiffus_t
    {
        using own_info_type = spade::omni::info_list_t<spade::omni::info::gradient, spade::omni::info::metric>;
        using omni_type = spade::omni::prefab::face_mono_t<own_info_type>;
        
        ldiffus_t(const float_t& coeff_in) : coeff{coeff_in} {}
        
        auto operator() (const auto& input) const
        {
            // don't forget the negative signs
            const auto& g = spade::omni::access<spade::omni::info::gradient>(input.face(0_c));
            const auto& n = spade::omni::access<spade::omni::info::metric>  (input.face(0_c));
            return -coeff*(g[0]*n[0] + g[1]*n[1]);
        }
        
        float_t coeff;
    };
}

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    std::vector<std::string> args;
    for (auto i: range(0, argc)) args.push_back(std::string(argv[i]));
    std::string input_filename = "input.sdf";
    if (args.size() < 2)
    {
        if (group.isroot()) print("Warning: defaulting to", input_filename);
    }
    else
    {
        input_filename = args[1];
    }
    
    scidf::node_t input;
    scidf::clargs_t clargs(argc, argv);
    scidf::read(input_filename, input, clargs);
    
    //==========================================================================
    //Kuya, Y., & Kawai, S. (2020). 
    //A stable and non-dissipative kinetic energy and entropy preserving (KEEP)
    //scheme for non-conforming block boundaries on Cartesian grids.
    //Computers and Fluids, 200. https://doi.org/10.1016/j.compfluid.2020.104427
    //
    // Equations 50, 52
    //
    //==========================================================================
    const real_t dt               = scidf::required<real_t>      (input["Config"]["dt"])       >> scidf::greater_than(0.0);
    const int nt_max              = scidf::required<int>         (input["Config"]["nt_max"])    >> scidf::greater_than(0);
    const int nt_skip             = scidf::required<int>         (input["Config"]["nt_skip"])   >> scidf::greater_than(0);
    const int checkpoint_skip     = scidf::required<int>         (input["Config"]["ck_skip"])   >> scidf::greater_than(0);
    const int nx                  = scidf::required<int>         (input["Config"]["nx_cell"])   >> (scidf::greater_than(4) && scidf::even);
    const int ny                  = scidf::required<int>         (input["Config"]["ny_cell"])   >> (scidf::greater_than(4) && scidf::even);
    const int nxb                 = scidf::required<int>         (input["Config"]["nx_blck"])   >> (scidf::greater_than(0));
    const int nyb                 = scidf::required<int>         (input["Config"]["ny_blck"])   >> (scidf::greater_than(0));
    const int nguard              = scidf::required<int>         (input["Config"]["nguard"])    >> scidf::greater_than(0);
    const real_t xmin             = scidf::required<real_t>      (input["Config"]["xmin"])      ;
    const real_t xmax             = scidf::required<real_t>      (input["Config"]["xmax"])      >> scidf::greater_than(xmin);
    const real_t ymin             = scidf::required<real_t>      (input["Config"]["ymin"])      ;
    const real_t ymax             = scidf::required<real_t>      (input["Config"]["ymax"])      >> scidf::greater_than(ymin);
    const bool do_output          = scidf::required<bool>        (input["Config"]["output"])    ;
    const int  lrefine            = scidf::required<bool>        (input["Config"]["lrefine"])   ;
    const std::string init_file   = scidf::required<std::string> (input["Config"]["init_file"]) >> (scidf::is_file || scidf::equals("none"));
    const bool   do_refine        = scidf::required<bool>        (input["Config"]["do_refine"]) ;
    const real_t c_diffuse        = scidf::required<real_t>      (input["Param"]["c_diffuse"])  ;
    const real_t xi               = scidf::required<real_t>      (input["Param"]["xi"])         ;
    const real_t c_ref            = scidf::required<real_t>      (input["Param"]["c_ref"])      ;
    const real_t h_ref            = scidf::required<real_t>      (input["Param"]["h_ref"])      ;
    const real_t kappa            = scidf::required<real_t>      (input["Param"]["kappa"])      ;
    const std::string geom_file   = scidf::required<std::string> (input["Param"]["geom"])       >> (scidf::is_file);
    
    spade::ctrs::array<int, 2> num_blocks(nxb, nyb);
    spade::ctrs::array<int, 2> cells_in_block(nx, ny);
    spade::ctrs::array<int, 2> exchange_cells(nguard, nguard);
    spade::bound_box_t<real_t, 2> bounds;
    bounds.min(0) =  xmin;
    bounds.max(0) =  xmax;
    bounds.min(1) =  ymin;
    bounds.max(1) =  ymax;
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    spade::amr::amr_blocks_t blocks(num_blocks, bounds);
    using refine_t = typename decltype(blocks)::refine_type;
    spade::ctrs::array<bool, 2> periodic = false;
    refine_t ref0  = {true,  true};
    
    local::vgeom_t<real_t> geom;
    geom.read(geom_file);
    if (do_refine) geom.refine_blks(lrefine, blocks, periodic, ref0);

    spade::grid::cartesian_grid_t grid(cells_in_block, exchange_cells, blocks, coords, group);
    auto handle = spade::grid::create_exchange(grid, group, periodic);
    
    real_t fill1 = 0.0;

    spade::grid::grid_array prim (grid, fill1);
    spade::grid::grid_array rhs  (grid, fill1);
    
    auto ini = _sp_lambda (const spade::coords::point_t<real_t>& x)
    {
        return 0.0;
    };

    spade::algs::fill_array(prim, ini);
    
    
    handle.exchange(prim);
    
    if (init_file != "none")
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_file, prim);
        if (group.isroot()) print("Init done.");
        handle.exchange(prim);
    }
    
    geom.create_sources(prim);
    
    // const auto s0 = spade::convective::cent_keep<4>(air);
    // spade::convective::rusanov_t       flx    (air);
    // spade::convective::weno_t          s1     (flx);
    // spade::state_sensor::ducros_t      ducr   (1.0e-2);
    // spade::convective::hybrid_scheme_t tscheme(s0, s1, ducr);
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    real_t time0 = 0.0;
    
    local::ldiffus_t diffus_flux(c_diffuse);
    
    auto calc_rhs = [&](auto& resid, const auto& sol, const auto& t)
    {
        resid = 0.0;
        spade::bound_box_t<bool, 2> bdy(false);
        spade::pde_algs::flux_div(sol, resid, bdy, diffus_flux);
        spade::pde_algs::source_term(sol, resid, [=](const real_t& val) {return -kappa*val;});
        geom.apply_source(sol, resid, h_ref, xi);
    };
    
    auto boundary_cond = [&](auto& sol, const auto& t)
    {
        handle.exchange(sol);
    };
    
    spade::time_integration::time_axis_t       axis(time0, dt);
    spade::time_integration::rk4_t             alg;
    spade::time_integration::integrator_data_t q(prim, rhs, alg);
    spade::time_integration::integrator_t      time_int(axis, alg, q, calc_rhs, boundary_cond);
    
    spade::timing::mtimer_t tmr("advance");
    for (auto nt: range(0, nt_max+1))
    {
        if (group.isroot())
        {
            const int pn = 10;
            print("nt: ",  spade::utils::pad_str(nt, pn));
        }
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "prims"+nstr;
            if (do_output) spade::io::output_vtk("output", filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = "checkpoint/"+filename+".bin";
            if (do_output) spade::io::binary_write(filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
    
    	tmr.start("advance");
        time_int.advance();
        tmr.stop("advance");
        if (group.isroot()) print(tmr);
    }
    return 0;
}
