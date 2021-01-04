#ifndef GENIF_TOOLS_H_
#define GENIF_TOOLS_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

namespace genif {
        /**
         * Provide different handy tools to accomplish general tasks in this framework.
         */
        class Tools {
        public:
            /**
             * Handles the worker count, which is passed to different functions. This function checks, whether the given worker count is less than one. In this case the maximum
             * number of available cores is returned. Otherwise, the passed workerCount is returned.
             * @param workerCount The worker count to "handle".
             * @return The "handled" worker count.
             */
            static unsigned int handleWorkerCount(int workerCount) {
                if (workerCount < 1)
                    return getWorkerCount();
                else
                    return workerCount;
            }

        private:
            /**
             * Returns the number of available processing cores by evaluating `std::thread::hardware_concurrency()`. If the function call fails, one is returned.
             * @return As stated above.
             */
            static unsigned int getWorkerCount() {
                const auto hw = std::thread::hardware_concurrency();
                return hw > 0 ? hw : 1;
            }
        };
    }

#endif /* GENIF_TOOLS_H_ */
