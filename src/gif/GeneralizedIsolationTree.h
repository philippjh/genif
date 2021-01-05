#ifndef GENIF_GENERALIZEDISOLATIONTREE_H
#define GENIF_GENERALIZEDISOLATIONTREE_H

#include "GIFExitCondition.h"
#include "GIFModel.h"
#include "Tree.h"
#include <chrono>
#include <nanoflann.hpp>
#include <random>
#include <src/Learner.h>
#include <src/OutlierDetectionResult.h>

namespace genif {
    class GeneralizedIsolationTree : public Learner<GIFModel, OutlierDetectionResult> {
    public:
        /**
         * Constructs an instance of GeneralizedIsolationTree.
         * @param k The number of representatives to find for each node.
         * @param exitCondition An exit condition, which controls, when tree induction is stopped.
         * @param workerCount Number of workers to consider.
         * @param seed Seed to use for random number generation (-1 defaults to sysclock seed). Pass an integer for constant result across multiple runs.
         */
        GeneralizedIsolationTree(unsigned int k, const GIFExitCondition& exitCondition, unsigned int workerCount, int seed = -1) :
            _k(k), _workerCount(workerCount), _exitCondition(exitCondition), _seed(seed) {
            if (_k <= 1)
                throw std::runtime_error("GeneralizedIsolationTree::GeneralizedIsolationTree: k needs to be at least two.");
            if (_workerCount < 1)
                throw std::runtime_error("GeneralizedIsolationTree::GeneralizedIsolationTree: workerCount needs to be at least one.");
        }

        /**
         * Fits the tree using a given dataset.
         * @param dataset The dataset to use for fitting.
         * @return A reference to this object.
         */
        Learner<GIFModel, OutlierDetectionResult>& fit(const MatrixX& dataset) override {
            // Check, whether we have enough observations.
            if (dataset.rows() < _k)
                throw std::runtime_error("GeneralizedIsolationTree::fit: The dataset should have at least k = " + std::to_string(_k) + " observations but has "
                                         + std::to_string(dataset.rows()) + " observations.");

            // Assign the tree to this object.
            Tree* treeRoot = findTree(dataset);

            // Find leafs.
            std::vector<unsigned int> leafVectorIndices;
            std::function<void(const Tree&)> findRepresentatives = [&findRepresentatives, &leafVectorIndices](const Tree& node) {
                if (!node.nodes.empty()) {
                    for (auto& childNode : node.nodes)
                        findRepresentatives(*childNode);
                } else
                    leafVectorIndices.push_back(node.representativeIndex);
            };
            findRepresentatives(*treeRoot);

            // Delete the tree since we do not need it anymore.
            delete treeRoot;

            // Create a GIFModel instance.
            GIFModel resultModel;

            // Build matrix from leaf nodes.
            resultModel.dataMatrix = std::make_shared<MatrixX>(leafVectorIndices.size(), dataset.cols());
            for (unsigned int i = 0; i < leafVectorIndices.size(); i++)
                resultModel.dataMatrix->row(i) = dataset.row(leafVectorIndices[i]);

            // Build KDTree on summary.
            auto kdTree = std::make_shared<nanoflann::KDTreeEigenMatrixAdaptor<MatrixX>>(resultModel.dataMatrix->cols(), std::cref(*resultModel.dataMatrix), 10);
            kdTree->index->buildIndex();

            // Iterate through the dataset and determine for each vector the nearest vectors in the summary.
            resultModel.countsPerRegion = std::vector<unsigned long>(resultModel.dataMatrix->rows(), 0);
#pragma omp parallel for num_threads(_workerCount)
            for (unsigned long i = 0; i < dataset.rows(); i++) {
                // Make KNN query for nearest summary vector.
                size_t nearestSummaryIndex;
                data_t sqDistance;
                nanoflann::KNNResultSet<data_t> resultSet(1);
                resultSet.init(&nearestSummaryIndex, &sqDistance);

                VectorX datasetVector = dataset.row(i);
                kdTree->index->findNeighbors(resultSet, datasetVector.data(), nanoflann::SearchParams(10));

                // Increase count for nearest summary point.
#pragma omp critical
                resultModel.countsPerRegion[nearestSummaryIndex] += 1;
            }

            // Calculate estimated probabilities for every region.
            resultModel.probabilitiesPerRegion = std::vector<data_t>(resultModel.dataMatrix->rows(), 0.0);
            for (unsigned long i = 0; i < resultModel.dataMatrix->rows(); i++)
                resultModel.probabilitiesPerRegion[i] = static_cast<data_t>(resultModel.countsPerRegion[i]) / static_cast<data_t>(dataset.size());

            // Assign properties.
            resultModel.dataKDTree = kdTree;
            _model = resultModel;

            return *this;
        }

        /**
         * Finds a tree using a given dataset.
         * @param dataset The dataset to create the tree from.
         * @return A raw pointer to the induced tree.
         */
        Tree* findTree(const MatrixX& dataset) {
            // Create PRNG.
            std::default_random_engine generator(_seed >= 0 ? _seed : std::chrono::system_clock::now().time_since_epoch().count());

            // Initialize a tree.
            Tree* treeRoot = new Tree(dataset);
            for (unsigned int i = 0; i < dataset.rows(); i++)
                treeRoot->vectorIndices.emplace_back(i);

            // Choose a (somewhat) random representative.
            treeRoot->representativeIndex = 0;

            // Create a worker data structure.
            std::vector<std::pair<unsigned int, Tree*>> treeTasks;
            treeTasks.emplace_back(0, treeRoot);

            while (!treeTasks.empty()) {
                // Choose a root node to work on.
                auto task = treeTasks.back();
                treeTasks.pop_back();

                unsigned int treeHeight = task.first;
                Tree* root = task.second;

                // Check, whether the exit condition already applies.
                bool shouldExit = _exitCondition.shouldExitRecursion(*root);
                if (!shouldExit) {
                    // Randomly sample representatives from node.
                    std::set<unsigned int> repIndices;
                    std::uniform_int_distribution<unsigned int> distribution(0, root->vectorIndices.size() - 1);
                    for (unsigned int j = 0; j < _k; j++) {
                        unsigned int nextIndex = root->vectorIndices[distribution(generator)];
                        while (repIndices.find(nextIndex) != repIndices.end())
                            nextIndex = root->vectorIndices[distribution(generator)];
                        repIndices.insert(nextIndex);
                    }
                    std::vector<unsigned int> clusterRepIndices(repIndices.begin(), repIndices.end());

                    // Generate clustering.
                    std::vector<std::vector<unsigned int>> clusters(clusterRepIndices.size());
#pragma omp parallel for num_threads(_workerCount)
                    for (unsigned int i = 0; i < root->vectorIndices.size(); i++) {
                        unsigned int fvIndex = root->vectorIndices[i];
                        unsigned int nearestIdx;
                        data_t nearestDist = std::numeric_limits<data_t>::max();

                        for (unsigned int j = 0; j < clusterRepIndices.size(); j++) {
                            data_t repDist = (dataset.row(fvIndex) - dataset.row(clusterRepIndices.at(j))).squaredNorm();
                            if (repDist < nearestDist) {
                                nearestDist = repDist;
                                nearestIdx = j;
                            }
                        }

                        // Put vector in bucket.
#pragma omp critical
                        clusters[nearestIdx].push_back(fvIndex);
                    }

                    // Every partition becomes a new node.
                    // Check, whether we have found exactly K clusters.
                    if (clusters.size() == _k) {
                        // Iterate all clusters and create new nodes from it.
                        for (unsigned int i = 0; i < clusters.size(); i++) {
                            // Create a new node.
                            Tree* node = new Tree(dataset);
                            node->vectorIndices = clusters[i];
                            node->representativeIndex = clusterRepIndices[i];
                            node->parent = root;

                            // Assign node to root.
                            root->nodes.push_back(node);

                            // If we have more than k observations in that node, we may create new tasks, which then are subject to further partitioning.
                            if (node->vectorIndices.size() > _k)
                                treeTasks.push_back(std::make_pair<unsigned int, Tree*>(treeHeight + 1, &*node));
                        }
                    } else
                        throw std::runtime_error("GeneralizedIsolationTree::fit: Clusterer did not return k = " + std::to_string(_k) + " clusters from "
                                                 + std::to_string(root->vectorIndices.size()) + " observations.");
                }
            }

            return treeRoot;
        }

        /**
         * Returns a previously fitted model.
         * @return As stated above.
         */
        GIFModel getModel() const override {
            return _model;
        }

        /**
         * Predicts the outlierness for a given dataset using a previously fitted model.
         * @param dataset The dataset to inspect for outliers.
         * @return An instance of OutlierDetectionResult which contains the probabilities for individual observations to be inliers.
         */
        OutlierDetectionResult predict(const MatrixX& dataset) const override {
            return predict(dataset, _model);
        }

        /**
         * Predicts the outlierness for a given dataset using a previously fitted model.
         * @param dataset The dataset to inspect for outliers.
         * @param model The model to use for prediction.
         * @return An instance of OutlierDetectionResult which contains the probabilities for individual observations to be inliers.
         */
        OutlierDetectionResult predict(const MatrixX& dataset, const GIFModel& model) const override {
            if (!model.probabilitiesPerRegion.empty()) {
                // Create a result model.
                OutlierDetectionResult result;
                result.probabilities = VectorX::Zero(dataset.rows());

                // Make the anomaly decision for every data point.
#pragma omp parallel for num_threads(_workerCount)
                for (unsigned long i = 0; i < dataset.rows(); i++) {
                    // Make KNN query for nearest summary vector.
                    size_t nearestSummaryIndex;
                    data_t sqDistance;
                    nanoflann::KNNResultSet<data_t> resultSet(1);
                    resultSet.init(&nearestSummaryIndex, &sqDistance);

                    VectorX datasetVector = dataset.row(i);
                    model.dataKDTree->index->findNeighbors(resultSet, datasetVector.data(), nanoflann::SearchParams(10));

                    // Assign probability values.
                    result.probabilities[i] = model.probabilitiesPerRegion[nearestSummaryIndex];
                }

                return result;
            } else
                throw std::runtime_error("GeneralizedIsolationTree:predict: No model has been learnt yet. Please call `fit` or `fitPredict` first.");
        }

        /**
         * Takes a copy of this object.
         * @return An unique_ptr pointing to a copy of this instance.
         */
        std::unique_ptr<Learner<GIFModel, OutlierDetectionResult>> copy() const override {
            return std::make_unique<GeneralizedIsolationTree>(_k, _exitCondition, _workerCount, _seed);
        }

    private:
        unsigned int _k = 10;
        unsigned int _workerCount = 1;
        int _seed;
        const GIFExitCondition& _exitCondition;
        GIFModel _model;
    };
}

#endif // GENIF_GENERALIZEDISOLATIONTREE_H
