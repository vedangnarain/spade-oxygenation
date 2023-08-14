#pragma once

namespace local
{
    static inline bool intersect(const auto& bbx, const auto& p0, const auto& p1)
    {
        using real_t = double;
        const auto lft_of = [&](const real_t x, const real_t y)
        {
            return ((p1[1]-p0[1])*x + (p0[0]-p1[0])*y + (p1[0]*p0[1] - p0[0]*p1[1])) >= 0;
        };
        int b0 = lft_of(bbx.min(0), bbx.min(1));
        int b1 = lft_of(bbx.min(0), bbx.max(1));
        int b2 = lft_of(bbx.max(0), bbx.min(1));
        int b3 = lft_of(bbx.max(0), bbx.max(1));
        int sum = b0 + b1 + b2 + b3;
        auto bm = (sum == 0) || (sum == 4);
        bool am = 
            (p0[0] > bbx.max(0) && p1[0] > bbx.max(0)) ||
            (p0[0] < bbx.min(0) && p1[0] < bbx.min(0)) ||
            (p0[1] > bbx.max(1) && p1[1] > bbx.max(1)) ||
            (p0[1] < bbx.min(1) && p1[1] < bbx.min(1));
        return !(bm|am);
    }
    
    template <typename data_t>
    struct vgeom_t
    {
        std::vector<spade::coords::point_t<data_t>> points;
        std::vector<spade::ctrs::array<int, 2>>     edges;
        std::vector<spade::grid::cell_idx_t>        indices;
        std::vector<int>                            edge_id;
        std::vector<data_t>                         h_l, h_l_vessel, d_l;
        std::vector<data_t>                         areas;
        std::vector<data_t>                         diameters;
        
        void read(const std::string& filename)
        {
            std::ifstream gf(filename);
            std::string line;
            std::getline(gf, line);
            while(true)
            {
                std::getline(gf, line);
                data_t x, y, z;
                std::stringstream iss(line);
                iss >> x;
                iss >> y;
                iss >> z;
                if (iss.fail()) break;
                points.push_back({x, y, z});
            }
            while(true)
            {
                std::getline(gf, line);
                int n0, n1;
                std::stringstream iss(line);
                iss >> n0;
                iss >> n1;
                if (iss.fail()) break;
                edges.push_back({n0, n1});
            }
            while(true)
            {
                std::getline(gf, line);
                data_t h_l_loc;
                std::stringstream iss(line);
                iss >> h_l_loc;
                if (iss.fail()) break;
                h_l_vessel.push_back(h_l_loc);
            }
            while(true)
            {
                std::getline(gf, line);
                data_t d_l;
                std::stringstream iss(line);
                iss >> d_l;
                if (iss.fail()) break;
                diameters.push_back(d_l);
            }
        }
        
        void create_sources(const auto& arr)
        {
            struct cmpr_t
            {
                bool operator()(const spade::grid::cell_idx_t& a, const spade::grid::cell_idx_t& b) const
                {
                    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
                }
            };
            std::map<spade::grid::cell_idx_t, int, cmpr_t> edge_ids_loc;
            spade::bound_box_t<int, 4> bds;
            const auto& grid = arr.get_grid();
            bds.min(0) = 0;
            bds.min(1) = 0;
            bds.min(2) = 0;
            bds.min(3) = 0;
            bds.max(0) = grid.get_num_cells(0);
            bds.max(1) = grid.get_num_cells(1);
            bds.max(2) = grid.get_num_cells(2);
            bds.max(3) = grid.get_num_local_blocks();
            spade::grid::cell_idx_t cidx;
            spade::algs::md_loop(cidx, bds, [&](const auto& idx)
            {
                const auto dx = grid.get_dx(0, idx.lb());
                const auto dy = grid.get_dx(1, idx.lb());
                using point_t = spade::coords::point_t<data_t>;
                int ct = 0;
                for (const auto& edge: edges)
                {
                    point_t x0 = points[edge[0]];
                    point_t x1 = points[edge[1]];
                    data_t src_trm = h_l_vessel[ct];
                    spade::bound_box_t<data_t, 2> bbx;
                    const auto xxx = grid.get_coords(idx);
                    bbx.min(0) = xxx[0] - 0.5*dx;
                    bbx.max(0) = xxx[0] + 0.5*dx;
                    bbx.min(1) = xxx[1] - 0.5*dy;
                    bbx.max(1) = xxx[1] + 0.5*dy;
                    if (intersect(bbx, x0, x1))
                    {
                        edge_ids_loc[idx] = ct;
                    }
                    ct++;
                }
            });
            for (const auto& p: edge_ids_loc)
            {
                const auto dx = grid.get_dx(0, p.first.lb());
                const auto dy = grid.get_dx(1, p.first.lb());
                indices.push_back(p.first);
                int id = p.second;
                h_l.push_back(h_l_vessel[id]);
                d_l.push_back(diameters[id]);
                areas.push_back(dx*dy);
            }
        }
        
        void refine_blks(const int lev, auto& blks, const auto per, const auto ref)
        {
            while (true)
            {
                const auto isect = [&](const auto& node)
                {
                    const auto& box = blks.get_block_box(node.tag);
                    if (spade::utils::max(node.level[0], node.level[1]) >= lev) return false;
                    for (const auto edge: edges)
                    {
                        if (intersect(box, points[edge[0]], points[edge[1]])) return true;
                    }
                    return false;
                };
                auto cur = blks.select(isect);
                if (cur.size() == 0) return;
                blks.refine(cur, per, ref, spade::amr::constraints::factor2);
            }
        }
        
        void apply_source(const auto& sol, auto& rhs, const auto& href, const auto& xi)
        {
            auto rv       = rhs.image();
            const auto sv = sol.image();
            for (int i = 0; i < indices.size(); ++i)
            {
                const auto idx = indices[i];
                rv(idx) += spade::consts::pi*d_l[i]*xi*((h_l[i]/href) - sv(idx))/areas[i];
            }
        }
    };
}