#ifndef PYGIF_TREE_H
#define PYGIF_TREE_H

#include <src/io/DataTypeHandling.h>

namespace pygif {
    struct Tree {
        // Tree structure.
        std::vector<Tree*> nodes;
        Tree* parent;

        // Tree data.
        const MatrixX& dataset;
        std::vector<unsigned int> vectorIndices;
        unsigned int representativeIndex;

        Tree(const MatrixX& dataset) : dataset(dataset) {
            // Constructor.
        }

        virtual ~Tree() {
            for (auto& node : nodes)
                delete node;
        }
    };
}

#endif // PYGIF_GIF_TREE_H
